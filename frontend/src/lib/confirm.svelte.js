// In-page replacement for window.confirm(), which some browsers and web views
// block silently: it then returns false without showing anything, and the
// action looks like it does nothing. Rendered by ConfirmDialog.svelte.

export const confirmState = $state({
  open: false,
  message: '',
  confirmLabel: 'OK',
  danger: false,
});

let resolvePending = null;

/**
 * Ask the user to confirm an action.
 * @param {string} message
 * @param {{ confirmLabel?: string, danger?: boolean }} [options]
 * @returns {Promise<boolean>} true if confirmed
 */
export function confirmDialog(message, { confirmLabel = 'OK', danger = false } = {}) {
  // A new question cancels any pending one.
  resolvePending?.(false);
  Object.assign(confirmState, { open: true, message, confirmLabel, danger });
  return new Promise((resolve) => {
    resolvePending = resolve;
  });
}

export function closeConfirm(confirmed) {
  confirmState.open = false;
  resolvePending?.(confirmed);
  resolvePending = null;
}
