<template>
  <canvas
    ref="canvasRef"
    class="h-full w-full cursor-grab active:cursor-grabbing"
    @dblclick="resetView"
    @mousedown="startDrag"
    @mouseleave="stopDrag"
    @wheel.prevent="zoom"
  />
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Agent, RolloutTrack, Scenario } from '@/types/world'
import { currentThemeId } from '@/utils/themeManager'

const props = defineProps<{
  scenario: Scenario | null
  frame: number
  showMap: boolean
  showGroundTruth: boolean
  showPredictions: boolean
  selectedAgentId?: number | null
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const zoomLevel = ref(8)
const panX = ref(0)
const panY = ref(0)
const dragging = ref(false)
const lastX = ref(0)
const lastY = ref(0)
let resizeObserver: ResizeObserver | undefined

type Theme = ReturnType<typeof getTheme>

function cssVar(name: string, fallback: string) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

function isDarkMode() {
  return document.documentElement.classList.contains('dark')
}

function getTheme() {
  const dark = isDarkMode()
  return {
    bg: cssVar('--background', dark ? '#141416' : '#ffffff'),
    fg: cssVar('--foreground', dark ? '#f1f5f9' : '#111827'),
    muted: cssVar('--muted-foreground', dark ? '#94a3b8' : '#6b7280'),
    grid: dark ? '#1e293b' : '#e5e7eb',
    mapOther: dark ? '#475569' : '#c0c0c0',
    mapSolid: dark ? '#e2e8f0' : '#111827',
    mapCurb: '#9370DB',
    mapStop: '#808080',
    centerline: dark ? '#94a3b8' : '#111827',
    controlled: '#3b82f6',
    sdc: '#f59e0b',
    target: '#22c55e',
    npc: dark ? '#64748b' : '#7f8c8d',
    pedestrian: '#5b9ea6',
    cyclist: '#2dd4bf',
    goal: '#f43f5e',
    collision: '#ef4444',
    offroad: '#f97316',
    historyStart: '#ffcf77',
    historyEnd: '#ff4100',
    future: '#e11d48',
    predStart: '#77eaff',
    predEnd: '#0055ff',
    rollout: '#38bdf8',
  }
}

const PREDICTION_MODE_COLORS = [
  ['#00e5ff', '#0057ff'],
  ['#ffcf33', '#ff6b00'],
  ['#a78bfa', '#7c3aed'],
  ['#34d399', '#059669'],
  ['#fb7185', '#e11d48'],
  ['#f472b6', '#be185d'],
] as const

function context(): CanvasRenderingContext2D | null {
  const canvas = canvasRef.value
  return canvas ? canvas.getContext('2d') : null
}

function resize() {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.floor(rect.width * dpr))
  const height = Math.max(1, Math.floor(rect.height * dpr))
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  const ctx = context()
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0)
  draw()
}

function initWorld(ctx: CanvasRenderingContext2D) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  ctx.translate(rect.width / 2, rect.height / 2)
  ctx.scale(zoomLevel.value, -zoomLevel.value)
  ctx.translate(panX.value, panY.value)
}

function resetView() {
  const scenario = props.scenario
  const canvas = canvasRef.value
  if (!scenario || !canvas) return
  const points: number[][] = []
  scenario.map.forEach((line) => line.points.forEach((point) => points.push(point)))
  scenario.agents.forEach((agent) => {
    agent.history.forEach((point) => points.push(point))
    agent.future.forEach((point) => points.push(point))
  })
  scenario.rollout?.forEach((track) => track.points.forEach((point) => points.push(point)))
  if (points.length === 0) return

  const xs = points.map((point) => point[0] ?? 0)
  const ys = points.map((point) => point[1] ?? 0)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const rect = canvas.getBoundingClientRect()
  const spanX = Math.max(40, maxX - minX)
  const spanY = Math.max(40, maxY - minY)
  zoomLevel.value = Math.max(0.8, Math.min(rect.width / spanX, rect.height / spanY) * 0.82)
  panX.value = -((minX + maxX) / 2)
  panY.value = -((minY + maxY) / 2)
  draw()
}

function drawGrid(ctx: CanvasRenderingContext2D, theme: Theme) {
  ctx.save()
  ctx.strokeStyle = theme.grid
  ctx.lineWidth = 0.5 / zoomLevel.value
  const step = 10
  const range = 2000
  for (let value = -range; value <= range; value += step) {
    ctx.beginPath()
    ctx.moveTo(value, -range)
    ctx.lineTo(value, range)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(-range, value)
    ctx.lineTo(range, value)
    ctx.stroke()
  }
  ctx.restore()
}

function drawWorldLine(ctx: CanvasRenderingContext2D, points: number[][], color: string, width: number, alpha = 1, dash: number[] = []) {
  if (points.length < 2) return
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.strokeStyle = color
  ctx.lineWidth = width / zoomLevel.value
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.setLineDash(dash.map((value) => value / zoomLevel.value))
  ctx.beginPath()
  points.forEach((point, index) => {
    const x = point[0] ?? 0
    const y = point[1] ?? 0
    if (index === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  })
  ctx.stroke()
  ctx.restore()
}

function mapStyle(type: number, theme: Theme) {
  const t = Math.round(type)
  if (t === 0 || t === 15 || t === 16) return { color: theme.mapCurb, width: 1.8, dash: [] }
  if (t === 6) return { color: theme.mapStop, width: 1.7, dash: [] }
  if (t === 3 || t === 8 || t === 9 || t === 10) return { color: theme.mapSolid, width: 1.5, dash: [6, 6] }
  if (t === 1 || t === 4) return { color: theme.centerline, width: 1.2, dash: [1.5, 4.5] }
  if (t === 2 || t === 5 || t === 7) return { color: theme.mapSolid, width: 1.5, dash: [] }
  return { color: theme.mapOther, width: 1.2, dash: [] }
}

function mixColor(a: string, b: string, t: number, alpha = 1) {
  const ah = a.startsWith('#') ? a.slice(1) : 'ffffff'
  const bh = b.startsWith('#') ? b.slice(1) : 'ffffff'
  const ar = parseInt(ah.slice(0, 2), 16), ag = parseInt(ah.slice(2, 4), 16), ab = parseInt(ah.slice(4, 6), 16)
  const br = parseInt(bh.slice(0, 2), 16), bg = parseInt(bh.slice(2, 4), 16), bb = parseInt(bh.slice(4, 6), 16)
  return `rgba(${Math.round(ar + (br - ar) * t)}, ${Math.round(ag + (bg - ag) * t)}, ${Math.round(ab + (bb - ab) * t)}, ${alpha})`
}

function drawGradientTrajectory(ctx: CanvasRenderingContext2D, points: number[][], startColor: string, endColor: string, width: number, alpha = 1) {
  if (points.length < 2) return
  for (let idx = 0; idx < points.length - 1; idx++) {
    const t = idx / Math.max(1, points.length - 2)
    drawWorldLine(ctx, [points[idx]!, points[idx + 1]!], mixColor(startColor, endColor, t, alpha), width, 1)
  }
  for (let idx = 0; idx < points.length; idx += Math.max(1, Math.floor(points.length / 12))) {
    drawWorldCircle(ctx, points[idx]!, mixColor(startColor, endColor, idx / Math.max(1, points.length - 1), alpha), 2.2 / zoomLevel.value)
  }
}

function drawWorldCircle(ctx: CanvasRenderingContext2D, point: number[], color: string, radius: number, stroke = false) {
  ctx.save()
  ctx.beginPath()
  ctx.arc(point[0] ?? 0, point[1] ?? 0, radius, 0, Math.PI * 2)
  if (stroke) {
    ctx.strokeStyle = color
    ctx.lineWidth = 2 / zoomLevel.value
    ctx.stroke()
  } else {
    ctx.fillStyle = color
    ctx.fill()
  }
  ctx.restore()
}

function agentColor(agent: Agent, theme: Theme, goalReached = false, collision = false, offroad = false): string {
  if (collision) return theme.collision
  if (offroad) return theme.offroad
  if (goalReached) return theme.target
  if (agent.controlled) return theme.controlled
  if (agent.sdc) return theme.sdc
  if (agent.target) return theme.target
  if (agent.type === 'pedestrian') return theme.pedestrian
  if (agent.type === 'cyclist') return theme.cyclist
  return theme.npc
}

function drawFlippedLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.save()
  ctx.scale(1, -1)
  ctx.fillStyle = color
  ctx.font = `${12 / zoomLevel.value}px sans-serif`
  ctx.fillText(text, x, -y)
  ctx.restore()
}

function drawScoreLabel(ctx: CanvasRenderingContext2D, text: string, point: number[], color: string) {
  const x = point[0] ?? 0
  const y = point[1] ?? 0
  ctx.save()
  ctx.scale(1, -1)
  ctx.font = `${11 / zoomLevel.value}px sans-serif`
  const padX = 4 / zoomLevel.value
  const padY = 3 / zoomLevel.value
  const width = ctx.measureText(text).width
  const height = 14 / zoomLevel.value
  ctx.fillStyle = isDarkMode() ? 'rgba(20, 20, 22, 0.78)' : 'rgba(255, 255, 255, 0.82)'
  ctx.strokeStyle = color
  ctx.lineWidth = 1 / zoomLevel.value
  ctx.beginPath()
  ctx.roundRect(x + 2 / zoomLevel.value, -y - height - 2 / zoomLevel.value, width + padX * 2, height, 4 / zoomLevel.value)
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = color
  ctx.fillText(text, x + 2 / zoomLevel.value + padX, -y - 2 / zoomLevel.value - padY)
  ctx.restore()
}

function isSelectedAgent(agentId: number) {
  return props.selectedAgentId === null || props.selectedAgentId === undefined || props.selectedAgentId === agentId
}

function rolloutState(track: RolloutTrack | undefined, frame: number) {
  if (!track || track.points.length === 0) return null
  const idx = Math.max(0, Math.min(frame, track.points.length - 1))
  return {
    position: track.points[idx]!,
    heading: track.headings?.[idx],
  }
}

function isGoalReached(track: RolloutTrack | undefined, frame: number) {
  if (!track?.goal_reached?.length) return false
  const idx = Math.max(0, Math.min(frame, track.goal_reached.length - 1))
  return track.goal_reached.slice(0, idx + 1).some(Boolean)
}

function hasCollision(track: RolloutTrack | undefined, frame: number) {
  if (!track?.collision?.length) return false
  const idx = Math.max(0, Math.min(frame, track.collision.length - 1))
  return Boolean(track.collision[idx])
}

function hasOffroad(track: RolloutTrack | undefined, frame: number) {
  if (!track?.offroad?.length) return false
  const idx = Math.max(0, Math.min(frame, track.offroad.length - 1))
  return Boolean(track.offroad[idx])
}

function drawAgentBox(
  ctx: CanvasRenderingContext2D,
  agent: Agent,
  theme: Theme,
  state?: { position: number[]; heading?: number } | null,
  goalReached = false,
  collision = false,
  offroad = false
) {
  const [x, y] = state?.position ?? agent.position
  const [lengthRaw, widthRaw] = agent.size ?? [4.5, 2.0]
  const length = Math.max(lengthRaw ?? 4.5, agent.type === 'pedestrian' ? 0.8 : 2.0)
  const width = Math.max(widthRaw ?? 2.0, agent.type === 'pedestrian' ? 0.6 : 1.0)
  const color = agentColor(agent, theme, goalReached, collision, offroad)

  ctx.save()
  ctx.translate(x ?? 0, y ?? 0)
  ctx.rotate(state?.heading ?? agent.heading)
  ctx.fillStyle = mixColor(color, color, 0, collision ? 0.62 : offroad ? 0.58 : goalReached ? 0.55 : agent.controlled || agent.sdc ? 0.42 : 0.22)
  ctx.strokeStyle = color
  const selected = props.selectedAgentId === agent.id
  ctx.lineWidth = (collision ? 4.2 : offroad ? 4.0 : goalReached ? 3.8 : selected ? 3.4 : agent.controlled || agent.sdc ? 2.6 : 1.6) / zoomLevel.value
  ctx.beginPath()
  ctx.rect(-length / 2, -width / 2, length, width)
  ctx.fill()
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(length / 2, 0)
  ctx.lineTo(length / 2 - Math.min(length * 0.28, 1.1), width * 0.28)
  ctx.lineTo(length / 2 - Math.min(length * 0.28, 1.1), -width * 0.28)
  ctx.closePath()
  ctx.fillStyle = color
  ctx.fill()
  ctx.restore()

  if (selected || agent.sdc || agent.controlled || agent.target) {
    drawFlippedLabel(ctx, `${agent.id}`, (x ?? 0) + width, (y ?? 0) + width, color)
  }
}

function drawPredictions(ctx: CanvasRenderingContext2D, scenario: Scenario, theme: Theme) {
  if (!props.showPredictions) return
  const predictions = scenario.predictions?.filter((prediction) => isSelectedAgent(prediction.agent_id)) ?? []
  const scores = predictions.map((prediction) => prediction.score).filter((score) => Number.isFinite(score))
  const minScore = scores.length ? Math.min(...scores) : 0
  const maxScore = scores.length ? Math.max(...scores) : 1
  const scoreSpan = Math.max(maxScore - minScore, 1e-6)
  predictions.forEach((prediction) => {
    const palette = PREDICTION_MODE_COLORS[prediction.mode % PREDICTION_MODE_COLORS.length] ?? [theme.predStart, theme.predEnd]
    const scoreRank = (prediction.score - minScore) / scoreSpan
    const confidence = Math.max(0.15, Math.min(1, prediction.score))
    const visualWeight = Math.max(confidence, scoreRank)
    const alpha = 0.28 + visualWeight * 0.68
    const width = 1.0 + visualWeight * 2.4
    drawGradientTrajectory(ctx, prediction.points, palette[0], palette[1], width, alpha)
    const end = prediction.points[prediction.points.length - 1]
    if (end) {
      drawWorldCircle(ctx, end, palette[1], (3.2 + visualWeight * 4.2) / zoomLevel.value)
      if (visualWeight > 0.45) {
        drawScoreLabel(ctx, prediction.score.toFixed(2), end, palette[1])
      }
    }
  })
  scenario.rollout?.filter((track) => isSelectedAgent(track.agent_id)).forEach((track) => {
    const points = track.points.slice(0, Math.max(2, props.frame + 1))
    drawWorldLine(ctx, points, theme.rollout, 2.6, 0.88)
    const last = points[points.length - 1]
    if (last) drawWorldCircle(ctx, last, theme.rollout, 4.5 / zoomLevel.value)
  })
}

function draw() {
  const ctx = context()
  const canvas = canvasRef.value
  if (!ctx || !canvas) return
  const rect = canvas.getBoundingClientRect()
  const theme = getTheme()
  ctx.save()
  ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0)
  ctx.fillStyle = theme.bg
  ctx.fillRect(0, 0, rect.width, rect.height)
  const scenario = props.scenario
  if (!scenario) {
    ctx.restore()
    return
  }

  initWorld(ctx)
  drawGrid(ctx, theme)
  if (props.showMap) {
    scenario.map.forEach((line) => {
      const style = mapStyle(line.type, theme)
      drawWorldLine(ctx, line.points, style.color, style.width, 0.95, style.dash)
    })
  }
  if (props.showGroundTruth) {
    scenario.agents
      .filter((agent) => isSelectedAgent(agent.id))
      .forEach((agent) => {
        const selected = props.selectedAgentId === agent.id
        drawWorldLine(ctx, agent.future, theme.future, selected ? 3.2 : agent.target || agent.sdc ? 2.4 : 1.8, selected ? 0.98 : 0.82, [5, 3])
        const end = agent.future[agent.future.length - 1]
        if (end) drawWorldCircle(ctx, end, theme.future, (selected ? 5.8 : 4.2) / zoomLevel.value)
      })
  }
  scenario.agents.forEach((agent) => {
    const selected = props.selectedAgentId === agent.id
    drawGradientTrajectory(
      ctx,
      agent.history,
      theme.historyStart,
      theme.historyEnd,
      selected ? 3.0 : 2.0,
      selected ? 1 : agent.sdc || agent.controlled ? 0.85 : 0.42
    )
  })
  drawPredictions(ctx, scenario, theme)
  const rolloutByAgent = new Map((scenario.rollout ?? []).map((track) => [track.agent_id, track]))
  scenario.rollout
    ?.filter((track) => track.controlled && track.goal)
    .forEach((track) => {
      const reached = isGoalReached(track, props.frame)
      if (reached) return
      const state = rolloutState(track, props.frame)
      if (state?.position) {
        drawWorldLine(ctx, [state.position, track.goal!], theme.goal, 1.5, 0.72, [3, 4])
      }
      drawWorldCircle(ctx, track.goal!, theme.goal, 8 / zoomLevel.value, true)
      drawWorldCircle(ctx, track.goal!, theme.goal, 2.8 / zoomLevel.value)
    })
  scenario.agents.forEach((agent) => {
    const track = rolloutByAgent.get(agent.id)
    drawAgentBox(
      ctx,
      agent,
      theme,
      rolloutState(track, props.frame),
      isGoalReached(track, props.frame),
      hasCollision(track, props.frame),
      hasOffroad(track, props.frame)
    )
  })
  ctx.restore()
}

function startDrag(event: MouseEvent) {
  dragging.value = true
  lastX.value = event.clientX
  lastY.value = event.clientY
}

function stopDrag() {
  dragging.value = false
}

function onMouseMove(event: MouseEvent) {
  if (!dragging.value) return
  panX.value += (event.clientX - lastX.value) / zoomLevel.value
  panY.value -= (event.clientY - lastY.value) / zoomLevel.value
  lastX.value = event.clientX
  lastY.value = event.clientY
  draw()
}

function onMouseUp() {
  dragging.value = false
}

function zoom(event: WheelEvent) {
  const factor = event.deltaY < 0 ? 1.1 : 0.9
  zoomLevel.value = Math.max(0.1, Math.min(200, zoomLevel.value * factor))
  draw()
}

function zoomIn() {
  zoomLevel.value = Math.max(0.1, Math.min(200, zoomLevel.value * 1.2))
  draw()
}

function zoomOut() {
  zoomLevel.value = Math.max(0.1, Math.min(200, zoomLevel.value / 1.2))
  draw()
}

watch(
  () => props.scenario?.id,
  async () => {
    await nextTick()
    resetView()
  }
)
watch(() => [props.frame, props.showMap, props.showGroundTruth, props.showPredictions, props.selectedAgentId, currentThemeId.value], draw)

onMounted(() => {
  window.addEventListener('resize', resize)
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  if (canvasRef.value) {
    resizeObserver = new ResizeObserver(() => resize())
    resizeObserver.observe(canvasRef.value)
  }
  resize()
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
  resizeObserver?.disconnect()
  resizeObserver = undefined
})

defineExpose({ resetView, resize, zoomIn, zoomOut })
</script>
