<script lang="ts" setup>
import type { HTMLAttributes } from 'vue'
import { nextTick, onMounted, onUnmounted, reactive, ref, Teleport, watch } from 'vue'
import { Icon } from '@iconify/vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const props = withDefaults(
    defineProps<{
        icon: string
        title: string
        floatingInit?: {
            top?: number
            left?: number
            width?: number
            height?: number
            minWidth?: number
            minHeight?: number
            maxHeight?: number
        }
        class?: HTMLAttributes['class']
        defaultFloating?: boolean
        forceFloating?: boolean
        closable?: boolean
    }>(),
    {
        floatingInit: () => ({ top: 120, left: 120, width: 560, height: 480, minWidth: 320, minHeight: 260, maxHeight: 720 }),
        defaultFloating: false,
        forceFloating: false,
        closable: true
    }
)

const emit = defineEmits<{ close: []; resize: [{ width: number; height: number }] }>()

const collapse = ref(false)
const floating = ref(!!props.defaultFloating || !!props.forceFloating)

const floatingRef = ref<HTMLElement | null>(null)
let dragging = false
let startX = 0
let startY = 0
let startTop = props.floatingInit.top ?? 120
let startLeft = props.floatingInit.left ?? 120

const pos = reactive({
    top: props.floatingInit.top ?? 120,
    left: props.floatingInit.left ?? 120,
    width: props.floatingInit.width ?? 560,
    height: props.floatingInit.height ?? 480
})

// ── Pop-out window state ──
const popout = ref(false)
const popoutContainer = ref<HTMLElement | null>(null)
let popoutWindow: Window | null = null
let styleObserver: MutationObserver | null = null

function cloneStylesToWindow(targetWin: Window) {
    const targetDoc = targetWin.document
    // Copy all <style> and <link rel="stylesheet"> tags
    document.querySelectorAll('style, link[rel="stylesheet"]').forEach((node) => {
        targetDoc.head.appendChild(node.cloneNode(true))
    })
    // Copy adopted stylesheets (used by some Vite/Tailwind setups)
    if (document.adoptedStyleSheets?.length) {
        targetDoc.adoptedStyleSheets = [...document.adoptedStyleSheets]
    }
    // Mirror html classes for dark-mode / theme tokens
    targetDoc.documentElement.className = document.documentElement.className

    // Watch for dynamically added styles (HMR, lazy component styles)
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node instanceof HTMLStyleElement || (node instanceof HTMLLinkElement && node.rel === 'stylesheet')) {
                    targetDoc.head.appendChild(node.cloneNode(true))
                }
            }
        }
    })
    observer.observe(document.head, { childList: true })
    return observer
}

function openPopout() {
    const w = pos.width
    const h = pos.height
    const win = window.open('', '', `width=${w},height=${h},resizable=yes,scrollbars=yes`)
    if (!win) return

    win.document.open()
    win.document.write('<!DOCTYPE html><html><head></head><body style="margin:0"></body></html>')
    win.document.close()
    win.document.title = props.title

    styleObserver = cloneStylesToWindow(win)

    const container = win.document.createElement('div')
    container.id = 'popout-root'
    container.className = 'bg-background text-foreground'
    container.style.cssText = 'height:100vh;display:flex;flex-direction:column'
    win.document.body.appendChild(container)

    popoutWindow = win
    popoutContainer.value = container
    popout.value = true
    floating.value = false

    win.addEventListener('beforeunload', () => {
        cleanupPopout(true)
    })
}

function cleanupPopout(restoreFloating = false) {
    styleObserver?.disconnect()
    styleObserver = null
    popout.value = false
    popoutContainer.value = null
    popoutWindow = null
    if (restoreFloating) {
        nextTick(() => {
            floating.value = true
        })
    }
}

function closePopout() {
    const win = popoutWindow
    cleanupPopout(false)
    win?.close()
    floating.value = true
}

// ── Drag / resize logic ──
function onMouseDown(e: MouseEvent) {
    if (!floating.value) return
    dragging = true
    startX = e.clientX
    startY = e.clientY
    const el = floatingRef.value
    if (!el) return
    const rect = el.getBoundingClientRect()
    startTop = rect.top
    startLeft = rect.left
    document.body.classList.add('select-none')
}

function onMouseMove(e: MouseEvent) {
    if (!dragging) return
    const dx = e.clientX - startX
    const dy = e.clientY - startY
    pos.top = Math.max(8, startTop + dy)
    pos.left = Math.max(8, startLeft + dx)
}

function onMouseUp() {
    dragging = false
    document.body.classList.remove('select-none')
    updateSizeFromRef()
}

function handleClose() {
    emit('close')
    if (!props.forceFloating) {
        floating.value = false
    }
}

function updateSizeFromRef() {
    if (!floatingRef.value) return
    const rect = floatingRef.value.getBoundingClientRect()
    pos.width = rect.width
    pos.height = rect.height
    emit('resize', { width: rect.width, height: rect.height })
}

onMounted(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    window.addEventListener('mouseup', updateSizeFromRef)
})
onUnmounted(() => {
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mouseup', onMouseUp)
    window.removeEventListener('mouseup', updateSizeFromRef)
    // Clean up pop-out window if component is destroyed
    if (popoutWindow) {
        popoutWindow.close()
        cleanupPopout()
    }
})

watch(floating, (v) => {
    if (v && floatingRef.value) {
        const r = floatingRef.value.getBoundingClientRect()
        pos.top = r.top || pos.top
        pos.left = r.left || pos.left
    }
})

watch(
    () => props.forceFloating,
    (v) => {
        if (v) floating.value = true
    }
)
</script>

<template>
    <!-- ── Inline (non-floating, non-popout) ── -->
    <Card v-if="!floating && !popout" v-auto-animate :class="cn('gap-4 pt-2', props.class)">
        <CardHeader class="px-3">
            <CardTitle class="flex items-center gap-2">
                <Icon class="text-primary size-5" :icon="icon" />
                {{ title }}
                <Button
                    class="ml-auto cursor-pointer rounded-full"
                    :disabled="props.forceFloating"
                    size="icon-sm"
                    variant="ghost"
                    @click="floating = true"
                >
                    <Icon class="size-5" icon="lucide:maximize-2" />
                </Button>
                <Button class="cursor-pointer rounded-full" size="icon-sm" variant="ghost" @click="collapse = !collapse">
                    <Icon class="size-5 transition-all" :class="collapse && 'rotate-180'" icon="lucide:chevron-down" />
                </Button>
            </CardTitle>
        </CardHeader>

        <div class="grid min-h-0 transition-[grid-template-rows] duration-200 ease-out" :style="{ gridTemplateRows: collapse ? '0fr' : '1fr' }">
            <div class="flex min-h-0 flex-col gap-4 overflow-hidden">
                <div v-if="!collapse && $slots.header" class="px-4">
                    <slot name="header" />
                </div>

                <CardContent class="scrollbar-none space-y-4 overflow-y-auto px-4">
                    <slot />
                </CardContent>

                <CardFooter v-if="$slots.footer" class="space-y-4">
                    <slot name="footer" />
                </CardFooter>
            </div>
        </div>
    </Card>

    <!-- ── Floating overlay (same window) ── -->
    <Transition appear name="floating-fade">
        <div
            v-if="floating && !popout"
            ref="floatingRef"
            class="bg-card fixed z-50 flex min-h-[260px] resize flex-col overflow-hidden rounded-lg border shadow-2xl"
            :style="{
                top: pos.top + 'px',
                left: pos.left + 'px',
                width: pos.width + 'px',
                height: pos.height + 'px',
                minWidth: (props.floatingInit.minWidth ?? 320) + 'px',
                minHeight: (props.floatingInit.minHeight ?? 260) + 'px',
                maxHeight: props.floatingInit.maxHeight ? props.floatingInit.maxHeight + 'px' : undefined
            }"
        >
            <div class="bg-background flex cursor-grab items-center justify-between border-b px-3 py-2" @mousedown="onMouseDown">
                <h3 class="m-0 flex items-center gap-2 font-semibold">
                    <Icon class="text-primary size-5" :icon="icon" />
                    {{ title }}
                </h3>
                <div class="flex items-center gap-2">
                    <Button class="h-8 w-8" size="icon" title="Pop out to new window" variant="ghost" @click="openPopout">
                        <Icon class="size-4" icon="lucide:external-link" />
                    </Button>
                    <Button v-if="props.closable" class="h-8 w-8" size="icon" variant="ghost" @click="handleClose">
                        <Icon class="size-4" icon="lucide:x" />
                    </Button>
                </div>
            </div>

            <div v-if="$slots.header" class="px-4 pt-3">
                <slot name="header" />
            </div>

            <div class="scrollbar-thin flex-1 space-y-4 overflow-y-auto p-4">
                <slot />
            </div>

            <div v-if="$slots.footer" class="flex items-center p-4">
                <slot name="footer" />
            </div>
        </div>
    </Transition>

    <!-- ── Pop-out: content teleported into a separate browser window ── -->
    <Teleport v-if="popout && popoutContainer" :to="popoutContainer">
        <div class="bg-background flex items-center justify-between border-b px-3 py-2">
            <h3 class="m-0 flex items-center gap-2 font-semibold">
                <Icon class="text-primary size-5" :icon="icon" />
                {{ title }}
            </h3>
            <Button class="h-8 w-8" size="icon" variant="ghost" @click="closePopout">
                <Icon class="size-4" icon="lucide:x" />
            </Button>
        </div>

        <div v-if="$slots.header" class="px-4 pt-3">
            <slot name="header" />
        </div>

        <div class="scrollbar-thin flex-1 space-y-4 overflow-y-auto p-4">
            <slot />
        </div>

        <div v-if="$slots.footer" class="flex items-center p-4">
            <slot name="footer" />
        </div>
    </Teleport>
</template>

<style scoped>
.floating-fade-enter-active,
.floating-fade-leave-active {
    transition:
        opacity 180ms ease,
        transform 180ms ease;
}

.floating-fade-enter-from,
.floating-fade-leave-to {
    opacity: 0;
    transform: translateY(6px) scale(0.985);
}
</style>
