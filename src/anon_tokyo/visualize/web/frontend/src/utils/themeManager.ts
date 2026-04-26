import { ref } from 'vue'
import { THEMES } from '@/utils/themeRegistry'

const STORAGE_KEY = 'hermes-ui-theme'
const LINK_ID = 'hermes-theme-css'

export const currentThemeId = ref<string>(getStoredThemeId())

function getLink(): HTMLLinkElement {
    let link = document.getElementById(LINK_ID) as HTMLLinkElement | null
    if (!link) {
        link = document.createElement('link')
        link.id = LINK_ID
        link.rel = 'stylesheet'
        document.head.appendChild(link)
    }
    return link
}

export function getStoredThemeId(): string {
    return localStorage.getItem(STORAGE_KEY) ?? 'hermes-default'
}

export function setStoredThemeId(themeId: string) {
    localStorage.setItem(STORAGE_KEY, themeId)
    currentThemeId.value = themeId
}

export function applyTheme(themeId: string) {
    const theme = THEMES.find((t) => t.id === themeId) ?? THEMES[0]!
    const link = getLink()
    link.href = theme.cssPath
    if (currentThemeId.value !== themeId) {
        currentThemeId.value = themeId
    }
}

function withoutTransitions(fn: () => void) {
    // During heavy canvas scenes, animating theme token changes can tank FPS.
    // Disable transitions for a couple frames so the swap is a single commit.
    document.documentElement.classList.add('no-theme-transition')
    try {
        fn()
    } finally {
        // Wait 2 rAFs so styles are applied before re-enabling transitions.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                document.documentElement.classList.remove('no-theme-transition')
            })
        })
    }
}

export function setThemeOptimized(themeId: string) {
    withoutTransitions(() => applyTheme(themeId))
}

// Call once on app startup.
export function initTheme() {
    const id = getStoredThemeId()
    withoutTransitions(() => applyTheme(id))

    // Listen for storage changes from other tabs
    window.addEventListener('storage', (e) => {
        const next = e.newValue
        if (e.key === STORAGE_KEY && typeof next === 'string') {
            withoutTransitions(() => applyTheme(next))
        }
    })
}
