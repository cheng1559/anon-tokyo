<script lang="ts" setup>
import { onMounted, onUnmounted } from 'vue'

interface Props {
    color?: string
    darkColor?: string
    dotSize?: number
    hoverSize?: number
    smoothness?: number
    trailMaxLength?: number
}

const props = withDefaults(defineProps<Props>(), {
    color: '#323232a6',
    darkColor: '#ffffffa6',
    dotSize: 10,
    hoverSize: 10,
    smoothness: 0.15,
    trailMaxLength: 10
})

let observer: MutationObserver | null = null
let isDark = document.documentElement.classList.contains('dark')

function getActiveColor(): string {
    return isDark ? props.darkColor : props.color
}

let canvas: HTMLCanvasElement | null = null
let context: CanvasRenderingContext2D | null = null
let animationFrame = 0
let width = 0
let height = 0
const cursor = { x: 0, y: 0 }
let hasPointerPosition = false
let isHovering = false
let hoverEffect = 0
let clickEffect = 0
const clickPoint = { x: 0, y: 0 }
let currentSize = 0
let interactiveElements: Element[] = []

// Trail history: stores past positions of the main dot
let trail: { x: number; y: number }[] = []

function lerp(start: number, end: number, factor: number): number {
    return start + (end - start) * factor
}

function parseColor(color: string): { r: number; g: number; b: number; a: number } {
    const hex = color.replace('#', '')
    if (hex.length === 8) {
        return {
            r: parseInt(hex.slice(0, 2), 16),
            g: parseInt(hex.slice(2, 4), 16),
            b: parseInt(hex.slice(4, 6), 16),
            a: parseInt(hex.slice(6, 8), 16) / 255
        }
    }
    return {
        r: parseInt(hex.slice(0, 2), 16),
        g: parseInt(hex.slice(2, 4), 16),
        b: parseInt(hex.slice(4, 6), 16),
        a: 1
    }
}

class Dot {
    position: { x: number; y: number }
    lag: number

    constructor(x: number, y: number, lag: number) {
        this.position = { x, y }
        this.lag = lag
    }

    moveTowards(x: number, y: number) {
        this.position.x += (x - this.position.x) / this.lag
        this.position.y += (y - this.position.y) / this.lag
    }
}

let dot: Dot | null = null

function onMouseMove(e: MouseEvent) {
    cursor.x = e.clientX
    cursor.y = e.clientY

    if (!hasPointerPosition) {
        hasPointerPosition = true
        if (dot) {
            dot.position.x = cursor.x
            dot.position.y = cursor.y
            trail = [{ x: cursor.x, y: cursor.y }]
        }
    }
}

function onWindowResize() {
    width = window.innerWidth
    height = window.innerHeight
    if (canvas) {
        canvas.width = width
        canvas.height = height
    }
}

function onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return
    clickPoint.x = e.clientX
    clickPoint.y = e.clientY
    clickEffect = 1
}

function onHoverEnter() {
    isHovering = true
}

function onHoverLeave() {
    isHovering = false
}

function updateDot() {
    if (!context || !dot) return

    context.clearRect(0, 0, width, height)

    if (!hasPointerPosition) return

    // Move main dot towards cursor
    dot.moveTowards(cursor.x, cursor.y)

    // Update current size
    const targetSize = isHovering ? props.hoverSize : props.dotSize
    currentSize = lerp(currentSize, targetSize, props.smoothness)

    // Smooth hover visual transition (prevents abrupt ring pop-in/out)
    const targetHoverEffect = isHovering ? 1 : 0
    hoverEffect = lerp(hoverEffect, targetHoverEffect, 0.18)

    // Click pulse fade-out
    clickEffect = lerp(clickEffect, 0, 0.2)

    // Record position into trail history
    trail.unshift({ x: dot.position.x, y: dot.position.y })
    if (trail.length > props.trailMaxLength) {
        trail.length = props.trailMaxLength
    }

    // Draw continuous tapered trail as smooth filled ribbon
    if (trail.length > 2) {
        const baseColor = parseColor(getActiveColor())

        // Catmull-Rom spline interpolation to produce a dense, smooth path
        function catmullRom(
            p0: { x: number; y: number },
            p1: { x: number; y: number },
            p2: { x: number; y: number },
            p3: { x: number; y: number },
            t: number
        ) {
            const t2 = t * t
            const t3 = t2 * t
            return {
                x: 0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
                y: 0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
            }
        }

        // Subdivide trail into a denser set of points
        const subdivisions = 3
        const smooth: { x: number; y: number }[] = []
        for (let i = 0; i < trail.length - 1; i++) {
            const p0 = trail[Math.max(i - 1, 0)]!
            const p1 = trail[i]!
            const p2 = trail[Math.min(i + 1, trail.length - 1)]!
            const p3 = trail[Math.min(i + 2, trail.length - 1)]!
            for (let s = 0; s < subdivisions; s++) {
                smooth.push(catmullRom(p0, p1, p2, p3, s / subdivisions))
            }
        }
        smooth.push(trail[trail.length - 1]!)

        // Build left and right edge points for a tapered ribbon
        const leftEdge: { x: number; y: number }[] = []
        const rightEdge: { x: number; y: number }[] = []

        for (let i = 0; i < smooth.length; i++) {
            const t = i / (smooth.length - 1) // 0 = head, 1 = tail
            // Ease out: taper more gently near the head, faster near the tail
            const easedT = t * t
            const halfWidth = currentSize * (1 - easedT)

            if (halfWidth < 0.2) break

            // Compute normal from neighbors
            let dx: number, dy: number
            if (i === 0) {
                const next = smooth[1]!
                const curr = smooth[0]!
                dx = next.x - curr.x
                dy = next.y - curr.y
            } else if (i === smooth.length - 1) {
                const curr = smooth[i]!
                const prev = smooth[i - 1]!
                dx = curr.x - prev.x
                dy = curr.y - prev.y
            } else {
                const next = smooth[i + 1]!
                const prev = smooth[i - 1]!
                dx = next.x - prev.x
                dy = next.y - prev.y
            }
            const len = Math.sqrt(dx * dx + dy * dy) || 1
            const nx = -dy / len
            const ny = dx / len

            const curr = smooth[i]!
            leftEdge.push({ x: curr.x + nx * halfWidth, y: curr.y + ny * halfWidth })
            rightEdge.push({ x: curr.x - nx * halfWidth, y: curr.y - ny * halfWidth })
        }

        // Draw filled ribbon with gradient from opaque to transparent
        if (leftEdge.length > 2) {
            const headPt = smooth[0]!
            const tailPt = smooth[Math.min(leftEdge.length - 1, smooth.length - 1)]!
            const gradient = context.createLinearGradient(headPt.x, headPt.y, tailPt.x, tailPt.y)
            gradient.addColorStop(0, `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${baseColor.a})`)
            gradient.addColorStop(1, `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, 0)`)

            context.beginPath()
            // Left edge forward — smooth curve through all points
            context.moveTo(leftEdge[0]!.x, leftEdge[0]!.y)
            for (let i = 1; i < leftEdge.length - 1; i++) {
                const curr = leftEdge[i]!
                const next = leftEdge[i + 1]!
                const cpx = (curr.x + next.x) / 2
                const cpy = (curr.y + next.y) / 2
                context.quadraticCurveTo(curr.x, curr.y, cpx, cpy)
            }
            const lastLeft = leftEdge[leftEdge.length - 1]!
            context.lineTo(lastLeft.x, lastLeft.y)
            // Right edge backward
            const lastRight = rightEdge[rightEdge.length - 1]!
            context.lineTo(lastRight.x, lastRight.y)
            for (let i = rightEdge.length - 2; i > 0; i--) {
                const curr = rightEdge[i]!
                const prev = rightEdge[i - 1]!
                const cpx = (curr.x + prev.x) / 2
                const cpy = (curr.y + prev.y) / 2
                context.quadraticCurveTo(curr.x, curr.y, cpx, cpy)
            }
            context.lineTo(rightEdge[0]!.x, rightEdge[0]!.y)
            context.closePath()
            context.fillStyle = gradient
            context.fill()
        }
    }

    if (trail.length > 0) {
        context.save()
        context.globalCompositeOperation = 'destination-out'
        context.beginPath()
        context.arc(dot.position.x, dot.position.y, currentSize * 1.05, 0, 2 * Math.PI)
        context.fill()
        context.restore()
    }

    if (hoverEffect > 0.01) {
        const baseColor = parseColor(getActiveColor())
        const eased = hoverEffect * hoverEffect * (3 - 2 * hoverEffect) // smoothstep

        // Soft halo for stronger hover feedback
        context.beginPath()
        context.arc(dot.position.x, dot.position.y, currentSize * (1.2 + 0.35 * eased), 0, 2 * Math.PI)
        context.fillStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${Math.min(baseColor.a * 0.35 * eased, 0.35)})`
        context.fill()

        // Crisp outer ring to make the state change obvious
        context.beginPath()
        context.arc(dot.position.x, dot.position.y, currentSize * (1.05 + 0.2 * eased), 0, 2 * Math.PI)
        context.strokeStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${Math.min(baseColor.a * 1.2 * eased, 1)})`
        context.lineWidth = Math.max(1.5, currentSize * (0.08 + 0.14 * eased))
        context.stroke()
    }

    if (clickEffect > 0.01) {
        const baseColor = parseColor(getActiveColor())
        const eased = 1 - (1 - clickEffect) * (1 - clickEffect)

        context.beginPath()
        context.arc(clickPoint.x, clickPoint.y, currentSize * (1.05 + (1 - eased) * 1.5), 0, 2 * Math.PI)
        context.strokeStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${Math.min(baseColor.a * eased * 0.9, 0.9)})`
        context.lineWidth = Math.max(1.5, currentSize * (0.22 * eased + 0.08))
        context.stroke()
    }

    // Draw main dot on top
    context.fillStyle = getActiveColor()
    context.beginPath()
    context.arc(dot.position.x, dot.position.y, currentSize, 0, 2 * Math.PI)
    context.fill()
    context.closePath()
}

function loop() {
    updateDot()
    animationFrame = requestAnimationFrame(loop)
}

const prefersReducedMotion = typeof window !== 'undefined' ? window.matchMedia('(prefers-reduced-motion: reduce)') : null

function init() {
    if (prefersReducedMotion?.matches) return

    width = window.innerWidth
    height = window.innerHeight
    cursor.x = width / 2
    cursor.y = height / 2
    hasPointerPosition = false

    canvas = document.createElement('canvas')
    context = canvas.getContext('2d')
    canvas.style.position = 'fixed'
    canvas.style.top = '0'
    canvas.style.left = '0'
    canvas.style.pointerEvents = 'none'
    canvas.style.zIndex = '9999'
    canvas.width = width
    canvas.height = height
    document.body.appendChild(canvas)

    dot = new Dot(width / 2, height / 2, 4)
    currentSize = props.dotSize
    hoverEffect = 0
    clickEffect = 0
    trail = []

    observer = new MutationObserver(() => {
        isDark = document.documentElement.classList.contains('dark')
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })

    interactiveElements = Array.from(document.querySelectorAll('a, button, [role="button"], input, textarea, select, label, [tabindex]'))
    interactiveElements.forEach((el) => {
        el.addEventListener('mouseenter', onHoverEnter)
        el.addEventListener('mouseleave', onHoverLeave)
    })

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mousedown', onMouseDown)
    window.addEventListener('resize', onWindowResize)
    loop()
}

function destroy() {
    canvas?.remove()
    canvas = null
    trail = []
    hasPointerPosition = false
    hoverEffect = 0
    clickEffect = 0
    observer?.disconnect()
    observer = null
    interactiveElements.forEach((el) => {
        el.removeEventListener('mouseenter', onHoverEnter)
        el.removeEventListener('mouseleave', onHoverLeave)
    })
    interactiveElements = []
    cancelAnimationFrame(animationFrame)
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mousedown', onMouseDown)
    window.removeEventListener('resize', onWindowResize)
}

onMounted(() => {
    if (prefersReducedMotion) {
        prefersReducedMotion.onchange = () => {
            if (prefersReducedMotion.matches) {
                destroy()
            } else {
                init()
            }
        }
    }
    init()
})

onUnmounted(() => {
    destroy()
})
</script>

<template>
    <slot />
</template>
