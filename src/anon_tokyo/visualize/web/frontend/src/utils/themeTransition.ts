/**
 * Helper to perform a View Transition with a circular clip-path effect.
 *
 * @param position The position { clientX, clientY } to start the transition from.
 * @param updateCallback A function that performs the actual DOM/state update (e.g. changing the theme).
 */
export function startThemeTransition(position: { clientX: number; clientY: number } | undefined, updateCallback: () => void | Promise<void>) {
    if (!document.startViewTransition) {
        updateCallback()
        return
    }

    const x = position?.clientX ?? window.innerWidth / 2
    const y = position?.clientY ?? window.innerHeight / 2

    const endRadius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y))

    const transition = document.startViewTransition(async () => {
        await updateCallback()
    })

    transition.ready.then(() => {
        const clipPath = [`circle(0px at ${x}px ${y}px)`, `circle(${endRadius}px at ${x}px ${y}px)`]

        document.documentElement.animate(
            {
                clipPath: clipPath
            },
            {
                duration: 300,
                easing: 'ease-in',
                pseudoElement: '::view-transition-new(root)'
            }
        )
    })
}
