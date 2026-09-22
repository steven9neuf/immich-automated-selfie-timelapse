//! Frame extraction from videos.
//!
//! Immich locates a person's face only on a video's preview frame. That face is
//! used as a reference: frames are sampled across the video with ffmpeg, the
//! person is re-identified on each one (dlib HOG detector + 128-d face encoding),
//! every match goes through the regular pipeline, and the best ones are kept.

use super::processing::{save_image, AssetProcessResult, OutputDirs};
use super::TimeIntervalTracker;
use crate::config::{Config, VideoFramesConfig};
use crate::immich_api::{Asset, FaceData, ImmichClient};
use crate::models::dlib_landmarks::{encoding_distance, FaceRect};
use crate::models::DlibLandmarks;
use crate::pipeline::{
    computed_keys, load_image_with_orientation, ComputedValue, Pipeline, PipelineContext,
    PipelineResult,
};
use crate::web::AtomicSkipStats;

use bytes::Bytes;
use image::RgbImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Longest side of extracted frames.
const FRAME_MAX_SIDE: u32 = 1920;
/// Longest side used for face detection: HOG is slow on large images, and faces
/// too small to be found at this size are too small for the timelapse anyway.
const DETECT_MAX_SIDE: u32 = 960;

/// A frame to run through the pipeline.
struct Candidate {
    bytes: Bytes,
    face: FaceData,
    /// Position in the video, in seconds (the preview counts as 0).
    offset_secs: f64,
}

/// Quality metrics of a frame that passed the pipeline.
#[derive(Debug, Clone, Copy, Default)]
struct FrameMetrics {
    offset_secs: f64,
    sharpness: Option<f32>,
    yaw: Option<f32>,
    ear: Option<f32>,
}

/// How a dlib detection box maps to an Immich face box, as fractions of the dlib
/// box size. Immich boxes cover the whole face, dlib's HOG boxes are tighter.
#[derive(Debug, Clone, Copy)]
struct BoxCalibration {
    dx1: f32,
    dy1: f32,
    dx2: f32,
    dy2: f32,
}

impl Default for BoxCalibration {
    fn default() -> Self {
        Self {
            dx1: -0.1,
            dy1: -0.3,
            dx2: 0.1,
            dy2: 0.05,
        }
    }
}

impl BoxCalibration {
    fn between(dlib: FaceRect, immich: (f32, f32, f32, f32)) -> Self {
        let (x1, y1, x2, y2) = rect_f32(dlib);
        let (w, h) = ((x2 - x1).max(1.0), (y2 - y1).max(1.0));
        Self {
            dx1: (immich.0 - x1) / w,
            dy1: (immich.1 - y1) / h,
            dx2: (immich.2 - x2) / w,
            dy2: (immich.3 - y2) / h,
        }
    }

    fn apply(&self, dlib: FaceRect) -> (f32, f32, f32, f32) {
        let (x1, y1, x2, y2) = rect_f32(dlib);
        let (w, h) = (x2 - x1, y2 - y1);
        (
            x1 + self.dx1 * w,
            y1 + self.dy1 * h,
            x2 + self.dx2 * w,
            y2 + self.dy2 * h,
        )
    }
}

/// The person as seen on the preview frame.
struct Reference {
    encoding: Vec<f64>,
    calibration: BoxCalibration,
}

fn rect_f32(r: FaceRect) -> (f32, f32, f32, f32) {
    (r.0 as f32, r.1 as f32, r.2 as f32, r.3 as f32)
}

fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let w = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let h = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let inter = w * h;
    let union = (a.2 - a.0) * (a.3 - a.1) + (b.2 - b.0) * (b.3 - b.1) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Detect faces on a downscaled copy, returning boxes in full-size coordinates.
fn detect_faces(dlib: &DlibLandmarks, img: &RgbImage) -> Result<Vec<FaceRect>, String> {
    let (w, h) = img.dimensions();
    let scale = (DETECT_MAX_SIDE as f32 / w.max(h) as f32).min(1.0);
    let small;
    let target = if scale < 1.0 {
        small = image::imageops::resize(
            img,
            ((w as f32 * scale) as u32).max(1),
            ((h as f32 * scale) as u32).max(1),
            image::imageops::FilterType::Triangle,
        );
        &small
    } else {
        img
    };
    let rects = dlib
        .detect_faces(
            target.width() as usize,
            target.height() as usize,
            target.as_raw(),
        )
        .map_err(|e| e.to_string())?;
    Ok(rects
        .into_iter()
        .map(|(x1, y1, x2, y2)| {
            (
                (x1 as f32 / scale) as i64,
                (y1 as f32 / scale) as i64,
                (x2 as f32 / scale) as i64,
                (y2 as f32 / scale) as i64,
            )
        })
        .collect())
}

/// Build the reference encoding from the preview, where Immich located the person.
fn reference_from_preview(preview: &[u8], face: &FaceData) -> Result<Reference, String> {
    let img = load_image_with_orientation(preview)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (w, h) = img.dimensions();

    // Immich box, rescaled to the decoded preview.
    let sx = w as f32 / face.image_width.max(1) as f32;
    let sy = h as f32 / face.image_height.max(1) as f32;
    let immich = (
        face.bounding_box_x1 * sx,
        face.bounding_box_y1 * sy,
        face.bounding_box_x2 * sx,
        face.bounding_box_y2 * sy,
    );

    // dlib's own box for the same face, if found: better landmarks for the
    // encoding, and the dlib -> Immich box mapping for this video.
    let dlib = DlibLandmarks::global().map_err(|e| e.to_string())?;
    let matched = detect_faces(dlib, &img)?
        .into_iter()
        .map(|r| (r, iou(rect_f32(r), immich)))
        .filter(|(_, overlap)| *overlap > 0.2)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(r, _)| r);

    let (rect, calibration) = match matched {
        Some(r) => (r, BoxCalibration::between(r, immich)),
        None => (
            (
                immich.0 as i64,
                immich.1 as i64,
                immich.2 as i64,
                immich.3 as i64,
            ),
            BoxCalibration::default(),
        ),
    };

    let encoding = dlib
        .face_encodings(w as usize, h as usize, img.as_raw(), &[rect])
        .map_err(|e| e.to_string())?
        .pop()
        .ok_or("no encoding for the reference face")?;
    Ok(Reference {
        encoding,
        calibration,
    })
}

/// Find the reference person on a frame. Returns their face box, Immich-style.
fn match_person(
    frame: &[u8],
    reference: &Reference,
    threshold: f32,
) -> Result<Option<FaceData>, String> {
    let img = image::load_from_memory(frame)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let dlib = DlibLandmarks::global().map_err(|e| e.to_string())?;

    let rects = detect_faces(dlib, &img)?;
    let encodings = dlib
        .face_encodings(w as usize, h as usize, img.as_raw(), &rects)
        .map_err(|e| e.to_string())?;

    let best = rects
        .into_iter()
        .zip(encodings)
        .map(|(r, e)| (r, encoding_distance(&e, &reference.encoding)))
        .min_by(|a, b| a.1.total_cmp(&b.1));

    Ok(best
        .filter(|(_, distance)| *distance <= threshold as f64)
        .map(|(r, _)| {
            let (x1, y1, x2, y2) = reference.calibration.apply(r);
            FaceData {
                bounding_box_x1: x1.max(0.0),
                bounding_box_y1: y1.max(0.0),
                bounding_box_x2: x2.min(w as f32),
                bounding_box_y2: y2.min(h as f32),
                image_width: w,
                image_height: h,
            }
        }))
}

/// Video duration in seconds.
async fn probe_duration(video: &Path) -> Result<f64, String> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(video)
        .output()
        .await
        .map_err(|e| format!("ffprobe: {}", e))?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|_| "ffprobe: unknown duration".to_string())
}

/// Extract one frame every `interval` seconds as JPEG files in `dir`.
async fn extract_frames(
    video: &Path,
    dir: &Path,
    interval: f64,
    max_frames: u32,
    cancel_token: &CancellationToken,
) -> Result<Vec<PathBuf>, String> {
    let filter = format!(
        "fps=fps={fps},scale='min({max},iw)':'min({max},ih)':force_original_aspect_ratio=decrease",
        fps = 1.0 / interval,
        max = FRAME_MAX_SIDE
    );
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(video)
        .args(["-vf", &filter, "-frames:v", &max_frames.to_string()])
        .args(["-q:v", "2"])
        .arg(dir.join("frame_%05d.jpg"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("ffmpeg: {}", e))?;

    let status = tokio::select! {
        status = child.wait() => status.map_err(|e| format!("ffmpeg: {}", e))?,
        _ = cancel_token.cancelled() => return Err("cancelled".to_string()),
    };
    if !status.success() {
        return Err(format!("ffmpeg exited with {}", status));
    }

    let mut frames = Vec::new();
    let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| e.to_string())?;
    while let Some(entry) = entries.next_entry().await.map_err(|e| e.to_string())? {
        let path = entry.path();
        let is_frame = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("frame_") && n.ends_with(".jpg"));
        if is_frame {
            frames.push(path);
        }
    }
    frames.sort();
    Ok(frames)
}

/// Sample the video and return the frames where the person was found, with the
/// video duration.
async fn sample_matching_frames(
    client: &ImmichClient,
    vf: &VideoFramesConfig,
    asset_id: &str,
    reference: Arc<Reference>,
    cancel_token: &CancellationToken,
) -> Result<(Vec<Candidate>, f64), String> {
    let dir = std::env::temp_dir().join(format!("immich-timelapse-{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| e.to_string())?;

    let result = async {
        let video = dir.join("video");
        client
            .download_video_to(asset_id, vf.use_original, &video)
            .await
            .map_err(|e| e.to_string())?;

        let duration = probe_duration(&video).await?;
        // Keep at most `max_candidates` frames: the interval widens on long videos.
        let interval = (vf.interval_secs as f64).max(duration / vf.max_candidates as f64);
        let frames =
            extract_frames(&video, &dir, interval, vf.max_candidates, cancel_token).await?;
        tracing::debug!(
            "Video {}: {:.1}s, {} frames sampled every {:.2}s",
            asset_id,
            duration,
            frames.len(),
            interval
        );

        let mut candidates = Vec::new();
        for (index, path) in frames.iter().enumerate() {
            if cancel_token.is_cancelled() {
                return Err("cancelled".to_string());
            }
            let bytes = Bytes::from(tokio::fs::read(path).await.map_err(|e| e.to_string())?);
            let reference = reference.clone();
            let threshold = vf.match_threshold;
            let frame = bytes.clone();
            let matched =
                tokio::task::spawn_blocking(move || match_person(&frame, &reference, threshold))
                    .await
                    .map_err(|e| e.to_string())?;
            match matched {
                Ok(Some(face)) => candidates.push(Candidate {
                    bytes,
                    face,
                    offset_secs: index as f64 * interval,
                }),
                Ok(None) => {}
                Err(e) => tracing::debug!("Video {} frame {}: {}", asset_id, index, e),
            }
        }
        Ok((candidates, duration))
    }
    .await;

    let _ = tokio::fs::remove_dir_all(&dir).await;
    result
}

/// Timestamp of a frame: the video's timestamp shifted by the frame offset.
/// Falls back to the video's timestamp when it cannot be parsed.
fn timestamp_at(base: &str, offset_secs: f64) -> String {
    if offset_secs <= 0.0 {
        return base.to_string();
    }
    match chrono::DateTime::parse_from_rfc3339(base) {
        Ok(t) => {
            let shifted = t + chrono::Duration::milliseconds((offset_secs * 1000.0) as i64);
            shifted
                .with_timezone(&chrono::Utc)
                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                .to_string()
        }
        Err(_) => base.to_string(),
    }
}

fn metrics(offset_secs: f64, computed: &HashMap<String, ComputedValue>) -> FrameMetrics {
    FrameMetrics {
        offset_secs,
        sharpness: computed
            .get(computed_keys::BLUR_METRIC)
            .and_then(|v| v.as_float()),
        yaw: computed
            .get(computed_keys::HEAD_POSE)
            .and_then(|v| v.as_head_pose())
            .map(|p| p.yaw),
        ear: computed.get(computed_keys::EAR).and_then(|v| v.as_float()),
    }
}

/// Pick up to `max` frames, best first, at least `min_gap` seconds apart.
///
/// Score = relative sharpness x frontality x eye openness; a metric whose step
/// is disabled counts as neutral.
fn select_best(frames: &[FrameMetrics], max: usize, min_gap: f64) -> Vec<usize> {
    let max_sharpness = frames
        .iter()
        .filter_map(|f| f.sharpness)
        .fold(0.0f32, f32::max);
    let score = |f: &FrameMetrics| {
        let sharp = match f.sharpness {
            Some(s) if max_sharpness > 0.0 => s / max_sharpness,
            _ => 1.0,
        };
        let frontal = f.yaw.map_or(1.0, |y| (1.0 - y.abs() / 90.0).max(0.0));
        let eyes = f.ear.map_or(1.0, |e| (e / 0.3).min(1.0));
        sharp * frontal * eyes
    };

    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&a, &b| score(&frames[b]).total_cmp(&score(&frames[a])));

    let mut picked: Vec<usize> = Vec::new();
    for i in order {
        if picked.len() >= max {
            break;
        }
        let far_enough = picked
            .iter()
            .all(|&p| (frames[p].offset_secs - frames[i].offset_secs).abs() >= min_gap);
        if far_enough {
            picked.push(i);
        }
    }
    picked
}

/// Process a video: sample frames, find the person, keep the best frames.
///
/// Falls back to the preview frame alone when the video cannot be sampled.
#[allow(clippy::too_many_arguments)]
pub async fn process_video_asset(
    client: &ImmichClient,
    config: &Config,
    asset: &Asset,
    face_data: &FaceData,
    timestamp: &str,
    output_dirs: &OutputDirs,
    cancel_token: &CancellationToken,
    skip_stats: &Arc<AtomicSkipStats>,
    pipeline: &Pipeline,
    time_interval: Option<&Arc<TimeIntervalTracker>>,
) -> AssetProcessResult {
    let vf = &config.processing.video_frames;
    let asset_id = asset.id.clone();

    // The preview is both the reference and the first candidate.
    let preview = match client.download_asset_preview(&asset_id).await {
        Ok(bytes) => bytes,
        Err(e) => {
            skip_stats.increment("download_failed");
            return AssetProcessResult::Error {
                asset_id,
                error: e.to_string(),
            };
        }
    };
    let mut candidates = vec![Candidate {
        bytes: preview.clone(),
        face: face_data.clone(),
        offset_secs: 0.0,
    }];
    let mut duration = 0.0;

    let reference = {
        let face = face_data.clone();
        tokio::task::spawn_blocking(move || reference_from_preview(&preview, &face))
            .await
            .unwrap_or_else(|e| Err(e.to_string()))
    };
    match reference {
        Ok(reference) => {
            match sample_matching_frames(client, vf, &asset_id, Arc::new(reference), cancel_token)
                .await
            {
                Ok((frames, d)) => {
                    candidates.extend(frames);
                    duration = d;
                }
                Err(e) => {
                    if cancel_token.is_cancelled() {
                        return AssetProcessResult::Cancelled { asset_id };
                    }
                    tracing::warn!("Video {}: {} - using the preview only", asset_id, e);
                }
            }
        }
        Err(e) => tracing::warn!(
            "Video {}: no reference face ({}) - using the preview only",
            asset_id,
            e
        ),
    }

    // Run every candidate through the regular pipeline. Per-frame skips are not
    // added to the job stats: the video counts once.
    let frame_stats = Arc::new(AtomicSkipStats::new());
    let debug_dir = output_dirs.debug.as_ref().map(|d| d.base.clone());
    let mut passed = Vec::new();
    let mut skip_reasons: HashMap<String, u32> = HashMap::new();
    for (index, candidate) in candidates.into_iter().enumerate() {
        if cancel_token.is_cancelled() {
            return AssetProcessResult::Cancelled { asset_id };
        }
        let frame_id = if index == 0 {
            asset_id.clone()
        } else {
            format!("{}_f{:03}", asset_id, index)
        };
        let frame_ts = timestamp_at(timestamp, candidate.offset_secs);
        let ctx = PipelineContext::new(frame_id.clone(), frame_ts.clone(), candidate.face)
            .with_bytes(candidate.bytes);
        match pipeline
            .execute(ctx, config, cancel_token, &frame_stats, debug_dir.as_ref())
            .await
        {
            PipelineResult::Success {
                image, computed, ..
            } => passed.push((
                frame_id,
                frame_ts,
                image,
                metrics(candidate.offset_secs, &computed),
            )),
            PipelineResult::Skipped { reason, .. } => {
                *skip_reasons.entry(reason).or_default() += 1;
            }
            PipelineResult::Error { error, .. } => {
                tracing::debug!("Video {} frame {}: {}", asset_id, index, error);
                *skip_reasons.entry("error".to_string()).or_default() += 1;
            }
            PipelineResult::Cancelled { .. } => {
                return AssetProcessResult::Cancelled { asset_id };
            }
        }
    }

    let max_frames = vf.max_frames_per_video as usize;
    let min_gap = (duration / (2.0 * max_frames as f64)).max(1.0);
    let frame_metrics: Vec<FrameMetrics> = passed.iter().map(|p| p.3).collect();
    let picked = select_best(&frame_metrics, max_frames, min_gap);

    let mut kept = 0;
    let mut time_slot_full = false;
    let mut passed: Vec<Option<_>> = passed.into_iter().map(Some).collect();
    for i in picked {
        let Some((frame_id, frame_ts, image, _)) = passed[i].take() else {
            continue;
        };
        if let Some(tracker) = time_interval {
            if !tracker.try_claim(&frame_ts) {
                time_slot_full = true;
                continue;
            }
        }
        if let Err(error) = save_image(image, &frame_ts, &frame_id, output_dirs).await {
            return AssetProcessResult::Error { asset_id, error };
        }
        skip_stats.increment_kept();
        kept += 1;
    }

    if kept > 0 {
        tracing::debug!("Video {}: kept {} frame(s)", asset_id, kept);
        return AssetProcessResult::Success { asset_id };
    }

    // Nothing kept: report the video once, under its most frequent reason.
    let reason = if time_slot_full {
        "time_interval".to_string()
    } else {
        skip_reasons
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(reason, _)| reason)
            .unwrap_or_else(|| "no_face_match".to_string())
    };
    skip_stats.increment(&reason);
    AssetProcessResult::Skipped { asset_id, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(offset_secs: f64, sharpness: f32, yaw: f32) -> FrameMetrics {
        FrameMetrics {
            offset_secs,
            sharpness: Some(sharpness),
            yaw: Some(yaw),
            ear: None,
        }
    }

    #[test]
    fn test_select_best_prefers_sharp_frontal_frames() {
        let frames = [
            frame(0.0, 50.0, 0.0),
            frame(5.0, 100.0, 0.0),
            frame(10.0, 100.0, 45.0),
        ];
        assert_eq!(select_best(&frames, 1, 1.0), vec![1]);
        assert_eq!(select_best(&frames, 2, 1.0), vec![1, 2]);
    }

    #[test]
    fn test_select_best_spreads_frames_in_time() {
        let frames = [
            frame(0.0, 100.0, 0.0),
            frame(0.5, 99.0, 0.0),
            frame(8.0, 60.0, 0.0),
        ];
        assert_eq!(select_best(&frames, 2, 2.0), vec![0, 2]);
    }

    #[test]
    fn test_select_best_neutral_when_metrics_missing() {
        let frames = [FrameMetrics::default(), FrameMetrics::default()];
        assert_eq!(select_best(&frames, 1, 1.0).len(), 1);
        assert!(select_best(&[], 3, 1.0).is_empty());
    }

    #[test]
    fn test_timestamp_at() {
        assert_eq!(
            timestamp_at("2024-01-15T12:34:56.000Z", 2.5),
            "2024-01-15T12:34:58.500Z"
        );
        assert_eq!(
            timestamp_at("2024-01-15T12:34:56.000Z", 0.0),
            "2024-01-15T12:34:56.000Z"
        );
        assert_eq!(timestamp_at("not a date", 3.0), "not a date");
    }

    #[test]
    fn test_box_calibration_roundtrip() {
        let dlib = (100, 100, 200, 200);
        let immich = (90.0, 70.0, 210.0, 205.0);
        let calibration = BoxCalibration::between(dlib, immich);
        let mapped = calibration.apply(dlib);
        assert!((mapped.0 - immich.0).abs() < 1e-3);
        assert!((mapped.1 - immich.1).abs() < 1e-3);
        assert!((mapped.2 - immich.2).abs() < 1e-3);
        assert!((mapped.3 - immich.3).abs() < 1e-3);
    }

    #[test]
    fn test_iou() {
        assert_eq!(iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)), 0.0);
        assert!((iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) - 1.0).abs() < 1e-6);
    }
}
