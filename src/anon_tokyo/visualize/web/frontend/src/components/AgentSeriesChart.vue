<script lang="ts" setup>
import { computed, ref, watch } from 'vue'
import { useColorMode } from '@vueuse/core'
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend)

const props = withDefaults(
  defineProps<{
    chartData: ChartData<'line'> | null
    frameIdx: number
    emptyText?: string
  }>(),
  {
    emptyText: 'No data'
  }
)

const chartRef = ref<typeof Line | null>(null)
const mode = useColorMode()

const currentFrameLine: Plugin<'line'> = {
  id: 'agentSeriesCurrentFrame',
  afterDraw(chart) {
    const { ctx, chartArea, scales } = chart
    if (!ctx || !chartArea || !scales.x) return

    const labels = chart.data.labels
    if (!labels || props.frameIdx < 0 || props.frameIdx >= labels.length) return

    const x = scales.x.getPixelForValue(props.frameIdx)
    ctx.save()
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(x, chartArea.top)
    ctx.lineTo(x, chartArea.bottom)
    ctx.stroke()
    ctx.restore()
  }
}

const chartOptions = computed<ChartOptions<'line'>>(() => {
  const isDark = mode.value === 'dark'
  const grid = isDark ? '#334155' : '#e5e7eb'
  const tick = isDark ? '#94a3b8' : '#64748b'
  const text = isDark ? '#f8fafc' : '#334155'

  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    spanGaps: false,
    elements: {
      line: { tension: 0.15 },
      point: { radius: 0, hitRadius: 8, hoverRadius: 3 }
    },
    scales: {
      x: {
        grid: { color: grid },
        ticks: { color: tick, maxRotation: 0, autoSkip: true },
        title: { display: false }
      },
      y: {
        grid: { color: grid },
        ticks: { color: tick },
        title: { display: false }
      }
    },
    plugins: {
      legend: {
        display: false,
        labels: { color: text }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        titleColor: text,
        bodyColor: text,
        backgroundColor: isDark ? '#0f172a' : '#f8fafc',
        borderColor: grid,
        borderWidth: 1
      }
    }
  }
})

watch(
  () => props.frameIdx,
  () => {
    const chart: ChartJS | undefined = chartRef.value?.chart
    if (!chart) return
    if (typeof chart.update === 'function') chart.update('none')
    else if (typeof chart.draw === 'function') chart.draw()
  }
)
</script>

<template>
  <div class="h-48 min-h-48 w-full rounded-md border p-2">
    <Line v-if="chartData" ref="chartRef" :data="chartData" :options="chartOptions" :plugins="[currentFrameLine]" />
    <div v-else class="text-muted-foreground flex h-full items-center justify-center text-sm">
      {{ emptyText }}
    </div>
  </div>
</template>
