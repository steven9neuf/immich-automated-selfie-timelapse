<script>
  import { confirmState, closeConfirm } from '../confirm.svelte.js';

  let cancelButton = $state(null);

  // Focus Cancel by default: most confirmations guard a deletion.
  $effect(() => {
    if (confirmState.open) cancelButton?.focus();
  });

  function handleKeydown(e) {
    if (confirmState.open && e.key === 'Escape') {
      e.preventDefault();
      closeConfirm(false);
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if confirmState.open}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div class="confirm-overlay" role="presentation" onclick={() => closeConfirm(false)}>
    <div
      class="confirm-dialog"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="confirm-message"
      tabindex="-1"
      onclick={(e) => e.stopPropagation()}
    >
      <p id="confirm-message">{confirmState.message}</p>
      <div class="confirm-actions">
        <button type="button" class="cancel-btn" bind:this={cancelButton} onclick={() => closeConfirm(false)}>
          Cancel
        </button>
        <button
          type="button"
          class="confirm-btn"
          class:danger={confirmState.danger}
          onclick={() => closeConfirm(true)}
        >
          {confirmState.confirmLabel}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .confirm-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    z-index: 1100;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
  }

  .confirm-dialog {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1.5rem;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  }

  .confirm-dialog p {
    margin: 0 0 1.5rem;
    white-space: pre-line;
    overflow-wrap: anywhere;
  }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
  }

  .confirm-actions button {
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    cursor: pointer;
  }

  .cancel-btn {
    background: #333;
    color: #e0e0e0;
  }

  .cancel-btn:hover {
    background: #444;
  }

  .confirm-btn {
    background: #4f46e5;
    color: #fff;
  }

  .confirm-btn:hover {
    background: #4338ca;
  }

  .confirm-btn.danger {
    background: #dc2626;
  }

  .confirm-btn.danger:hover {
    background: #b91c1c;
  }
</style>
