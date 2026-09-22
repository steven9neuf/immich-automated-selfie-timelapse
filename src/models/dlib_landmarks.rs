//! Dlib landmark predictor wrapper.
//!
//! Wraps the dlib-face-recognition crate's LandmarkPredictor and FaceDetector
//! in a thread-safe singleton to avoid reloading the model for every image.

use crate::error::{Error, Result};
use crate::pipeline::{Landmarks, Point};
use dlib_face_recognition::{
    FaceDetector, FaceDetectorTrait, FaceEncoderNetwork, FaceEncoderTrait, ImageMatrix,
    LandmarkPredictor, LandmarkPredictorTrait, Rectangle,
};
use std::sync::{Mutex, OnceLock};

/// Global landmark predictor instance.
/// Loaded lazily on first use.
static LANDMARK_PREDICTOR: OnceLock<Result<DlibLandmarks>> = OnceLock::new();

/// Face encoder (128-d ResNet descriptor), only needed for video frame extraction.
/// Loaded lazily and separately, so a missing model never breaks landmark detection.
static FACE_ENCODER: OnceLock<Result<FaceEncoder>> = OnceLock::new();

const FACE_ENCODER_MODEL: &str = "models/dlib_face_recognition_resnet_model_v1.dat";

struct FaceEncoder(Mutex<FaceEncoderNetwork>);

// Safety: same reasoning as DlibLandmarks, all access goes through the Mutex.
unsafe impl Send for FaceEncoder {}
unsafe impl Sync for FaceEncoder {}

impl FaceEncoder {
    fn global() -> Result<&'static FaceEncoder> {
        FACE_ENCODER
            .get_or_init(|| {
                FaceEncoderNetwork::open(FACE_ENCODER_MODEL)
                    .map(|net| FaceEncoder(Mutex::new(net)))
                    .map_err(|e| {
                        Error::Model(format!(
                            "Failed to load face encoder {}: {} \
                             (download http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)",
                            FACE_ENCODER_MODEL, e
                        ))
                    })
            })
            .as_ref()
            .map_err(|e| Error::Model(e.to_string()))
    }
}

/// Face rectangle in pixel coordinates: (x1, y1, x2, y2).
pub type FaceRect = (i64, i64, i64, i64);

/// Euclidean distance between two face encodings (same person below ~0.6).
pub fn encoding_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Thread-safe wrapper for dlib's FaceDetector and LandmarkPredictor.
///
/// The dlib types are not thread-safe, so we wrap them in a Mutex.
/// The model is loaded once and reused for all subsequent calls.
pub struct DlibLandmarks {
    detector: Mutex<FaceDetector>,
    predictor: Mutex<LandmarkPredictor>,
}

// Safety: The Mutex ensures exclusive access to the inner dlib types (FaceDetector and
// LandmarkPredictor), which are not Send/Sync themselves. All access goes through
// Mutex::lock(), guaranteeing only one thread uses them at a time.
unsafe impl Send for DlibLandmarks {}
unsafe impl Sync for DlibLandmarks {}

impl DlibLandmarks {
    /// Load the landmark predictor model from the given path.
    fn load() -> Result<Self> {
        let model_path = std::path::Path::new("models/shape_predictor_68_face_landmarks.dat");
        if !model_path.exists() {
            return Err(Error::Model(format!(
                "Landmark model not found at {}. \
                 Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                model_path.display()
            )));
        }

        let detector = FaceDetector::default();
        let predictor = LandmarkPredictor::open(model_path)
            .map_err(|e| Error::Model(format!("Failed to load landmark predictor: {}", e)))?;

        Ok(Self {
            detector: Mutex::new(detector),
            predictor: Mutex::new(predictor),
        })
    }

    /// Get or initialize the global landmark predictor instance.
    pub fn global() -> Result<&'static DlibLandmarks> {
        LANDMARK_PREDICTOR
            .get_or_init(DlibLandmarks::load)
            .as_ref()
            .map_err(|e| Error::Model(e.to_string()))
    }

    /// Eagerly initialize the model at startup.
    ///
    /// Call this during server startup to load the model before processing begins.
    /// This ensures any model loading messages appear during startup rather than
    /// during image processing.
    pub fn init() -> Result<()> {
        Self::global()?;
        Ok(())
    }

    /// Detect 68 facial landmarks from a cropped face image.
    ///
    /// # Arguments
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `pixels` - Raw RGB pixel data (width * height * 3 bytes)
    /// * `face_rect` - Optional face bounding box (x1, y1, x2, y2) in image coordinates.
    ///   If provided, uses this rectangle for landmark detection.
    ///   If None, attempts to detect the face or uses the whole image.
    ///
    /// # Returns
    /// Landmarks struct containing the 68 facial landmark points, or an error
    /// if landmarks could not be detected.
    pub fn detect_landmarks(
        &self,
        width: usize,
        height: usize,
        pixels: &[u8],
        face_rect: Option<(i64, i64, i64, i64)>,
    ) -> Result<Landmarks> {
        // Verify buffer is large enough for the given dimensions
        assert!(
            pixels.len() >= width * height * 3,
            "pixel buffer too small: need {} bytes for {}x{} RGB, got {}",
            width * height * 3,
            width,
            height,
            pixels.len()
        );

        // Create image matrix for dlib
        // Safety: We verified above that `pixels` has at least width*height*3 bytes,
        // matching the RGB layout that ImageMatrix::new expects.
        let matrix = unsafe { ImageMatrix::new(width, height, pixels.as_ptr()) };

        // Lock detector and predictor
        let detector = self
            .detector
            .lock()
            .map_err(|e| Error::Model(format!("Failed to lock detector: {}", e)))?;
        let predictor = self
            .predictor
            .lock()
            .map_err(|e| Error::Model(format!("Failed to lock predictor: {}", e)))?;

        // Determine the face rectangle to use
        let rect = if let Some((x1, y1, x2, y2)) = face_rect {
            // Use the provided face rectangle
            Rectangle {
                left: x1.max(0),
                top: y1.max(0),
                right: x2.min(width as i64),
                bottom: y2.min(height as i64),
            }
        } else {
            // No face rect provided - try to detect or use whole image
            let faces = detector.face_locations(&matrix);
            if !faces.is_empty() {
                faces[0]
            } else {
                // Fallback: use whole image with small margin
                let margin = 5;
                Rectangle {
                    left: margin,
                    top: margin,
                    right: (width as i64) - margin,
                    bottom: (height as i64) - margin,
                }
            }
        };

        // Detect landmarks
        let landmarks_raw = predictor.face_landmarks(&matrix, &rect);

        // Convert dlib landmarks to our Landmarks type
        let points: Vec<Point> = landmarks_raw
            .iter()
            .map(|p| Point::new(p.x() as f32, p.y() as f32))
            .collect();

        Landmarks::new(points).map_err(Error::Model)
    }

    /// Detect all faces in an RGB image (HOG detector).
    pub fn detect_faces(
        &self,
        width: usize,
        height: usize,
        pixels: &[u8],
    ) -> Result<Vec<FaceRect>> {
        assert!(pixels.len() >= width * height * 3, "pixel buffer too small");
        // Safety: buffer size checked above, RGB layout as expected by ImageMatrix::new.
        let matrix = unsafe { ImageMatrix::new(width, height, pixels.as_ptr()) };
        let detector = self
            .detector
            .lock()
            .map_err(|e| Error::Model(format!("Failed to lock detector: {}", e)))?;
        Ok(detector
            .face_locations(&matrix)
            .iter()
            .map(|r| (r.left, r.top, r.right, r.bottom))
            .collect())
    }

    /// Compute one 128-d face encoding per face rectangle of an RGB image.
    pub fn face_encodings(
        &self,
        width: usize,
        height: usize,
        pixels: &[u8],
        rects: &[FaceRect],
    ) -> Result<Vec<Vec<f64>>> {
        assert!(pixels.len() >= width * height * 3, "pixel buffer too small");
        if rects.is_empty() {
            return Ok(Vec::new());
        }
        let encoder = FaceEncoder::global()?;
        // Safety: buffer size checked above, RGB layout as expected by ImageMatrix::new.
        let matrix = unsafe { ImageMatrix::new(width, height, pixels.as_ptr()) };

        let landmarks: Vec<_> = {
            let predictor = self
                .predictor
                .lock()
                .map_err(|e| Error::Model(format!("Failed to lock predictor: {}", e)))?;
            rects
                .iter()
                .map(|&(x1, y1, x2, y2)| {
                    let rect = Rectangle {
                        left: x1.max(0),
                        top: y1.max(0),
                        right: x2.min(width as i64),
                        bottom: y2.min(height as i64),
                    };
                    predictor.face_landmarks(&matrix, &rect)
                })
                .collect()
        };

        let net = encoder
            .0
            .lock()
            .map_err(|e| Error::Model(format!("Failed to lock face encoder: {}", e)))?;
        Ok(net
            .get_face_encodings(&matrix, &landmarks, 0)
            .iter()
            .map(|e| e.as_ref().to_vec())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_distance() {
        assert_eq!(encoding_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        assert_eq!(encoding_distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_global_initialization() {
        // Just test that we can get the global instance
        // (this will fail if model file is not present, which is expected in CI)
        let _ = DlibLandmarks::global();
    }
}
