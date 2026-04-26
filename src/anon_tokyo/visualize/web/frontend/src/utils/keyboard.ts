import { onMounted, onUnmounted } from 'vue'

/**
 * Check whether the keyboard event originates from an editable element
 * (input, textarea, select, contenteditable) so that shortcuts don't
 * hijack normal text entry.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null
    if (!el) return false

    if (el.isContentEditable) return true

    const tag = el.tagName?.toLowerCase()
    if (!tag) return false

    if (tag === 'input' || tag === 'textarea' || tag === 'select') return true

    // Walk up: the actual target may be inside an editable wrapper.
    if (typeof el.closest === 'function') {
        if (el.closest('input, textarea, select, [contenteditable="true"]')) return true
    }

    return false
}

/**
 * Return true when a keyboard event should be ignored by app-level shortcuts
 * (e.g. the user is typing, or a modifier key is held that implies a browser /
 * OS shortcut rather than an app shortcut).
 */
export function shouldIgnoreKeyEvent(e: KeyboardEvent): boolean {
    if (isEditableTarget(e.target)) return true
    if (e.metaKey || e.ctrlKey || e.altKey) return true
    return false
}

export type KeyBinding = {
    /** One or more key values (KeyboardEvent.key) that trigger this binding. */
    keys: string | string[]
    /** The handler to invoke. Return nothing; preventDefault is called automatically. */
    handler: (e: KeyboardEvent) => void
    /** If true, `stopPropagation` is also called (useful for capture-phase listeners). */
    stopPropagation?: boolean
}

export interface UseKeyboardShortcutsOptions {
    /** Key bindings to register. */
    bindings: KeyBinding[]
    /**
     * If true, register on `document` with `{ capture: true }` instead of
     * `window` — useful for page-level shortcuts that must fire before
     * component-level ones.
     */
    capture?: boolean
    /**
     * Extra guard that runs *after* the default `shouldIgnoreKeyEvent` check.
     * Return true to skip handling.
     */
    extraGuard?: (e: KeyboardEvent) => boolean
}

/**
 * Vue composable that registers global keyboard shortcuts on mount and
 * removes them on unmount.  All bindings automatically skip editable
 * elements and modifier-key combos.
 *
 * @example
 * ```ts
 * useKeyboardShortcuts({
 *   bindings: [
 *     { keys: [' '], handler: () => togglePlayPause() },
 *     { keys: ['ArrowLeft', 'h', 'H'], handler: () => stepBackward() },
 *   ],
 * })
 * ```
 */
export function useKeyboardShortcuts(options: UseKeyboardShortcutsOptions) {
    const { bindings, capture = false, extraGuard } = options

    // Build a lookup map: key -> binding (for O(1) dispatch).
    const keyMap = new Map<string, KeyBinding>()
    for (const binding of bindings) {
        const keys = Array.isArray(binding.keys) ? binding.keys : [binding.keys]
        for (const key of keys) {
            keyMap.set(key, binding)
        }
    }

    function onKeydown(e: KeyboardEvent) {
        if (shouldIgnoreKeyEvent(e)) return
        if (extraGuard?.(e)) return

        const binding = keyMap.get(e.key)
        if (!binding) return

        e.preventDefault()
        if (binding.stopPropagation) e.stopPropagation()
        binding.handler(e)
    }

    const target: EventTarget = capture ? document : window
    const listenerOptions = capture ? { capture: true } : undefined

    onMounted(() => {
        target.addEventListener('keydown', onKeydown as EventListener, listenerOptions)
    })

    onUnmounted(() => {
        target.removeEventListener('keydown', onKeydown as EventListener, listenerOptions)
    })
}
