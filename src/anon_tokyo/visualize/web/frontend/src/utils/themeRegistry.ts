export type ThemeEntry = {
    id: string
    label: string
    cssPath: string
}

export const THEMES: ThemeEntry[] = [
    {
        id: 'hermes-default',
        label: 'Hermes Default',
        cssPath: new URL('../assets/themes/hermes-default.css', import.meta.url).toString()
    },
    {
        id: 'perpetuity',
        label: 'Perpetuity',
        cssPath: new URL('../assets/themes/perpetuity.css', import.meta.url).toString()
    },
    {
        id: 'claude',
        label: 'Claude',
        cssPath: new URL('../assets/themes/claude.css', import.meta.url).toString()
    },
    {
        id: 'claymorphism',
        label: 'Claymorphism',
        cssPath: new URL('../assets/themes/claymorphism.css', import.meta.url).toString()
    },
    {
        id: 'twitter',
        label: 'Twitter',
        cssPath: new URL('../assets/themes/twitter.css', import.meta.url).toString()
    },
    {
        id: 'supabase',
        label: 'Supabase',
        cssPath: new URL('../assets/themes/supabase.css', import.meta.url).toString()
    }
]
