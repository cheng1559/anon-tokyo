<script lang="ts" setup>
import 'vue-sonner/style.css'

import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Icon } from '@iconify/vue'
import { useColorMode } from '@vueuse/core'
import type { ChartData } from 'chart.js'
import { toast } from 'vue-sonner'
import AgentSeriesChart from '@/components/AgentSeriesChart.vue'
import Card from '@/components/card/Card.vue'
import CheckpointSelector from '@/components/checkpoint-selector/CheckpointSelector.vue'
import Footer from '@/components/footer/Footer.vue'
import ScenarioCanvas from '@/components/ScenarioCanvas.vue'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Spinner } from '@/components/ui/spinner'
import { Toaster } from '@/components/ui/sonner'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { fetchBatch, fetchEnv, fetchFiles, initializeEnv } from '@/api/world'
import type { Agent, BatchPayload, EnvInfo, FileCatalog, RolloutTrack } from '@/types/world'
import { currentThemeId, setStoredThemeId, setThemeOptimized } from '@/utils/themeManager'
import { THEMES } from '@/utils/themeRegistry'
import { startThemeTransition } from '@/utils/themeTransition'

const env = ref<EnvInfo | null>(null)
const files = ref<FileCatalog | null>(null)
const batch = ref<BatchPayload | null>(null)
const scenarioCanvas = ref<InstanceType<typeof ScenarioCanvas> | null>(null)
const batchIndex = ref(0)
const worldIndex = ref(0)
const task = ref('prediction')
const configPath = ref('configs/prediction/anon_tokyo.yaml')
const checkpointPath = ref('None')
const split = ref('validation')
const batchSize = ref(4)
const frame = ref(0)
const frameSlider = computed({
    get: () => [frame.value],
    set: (value: number[]) => {
        frame.value = value[0] ?? 0
    }
})
const playbackSpeeds = ref([10])
const playbackSpeed = computed(() => playbackSpeeds.value[0] ?? 10)
const playing = ref(false)
const showMap = ref(true)
const showGroundTruth = ref(true)
const showPredictions = ref(true)
const selectedAgentId = ref<number | null>(null)
const agentSignalTab = ref('reward')
const isLoading = ref(false)
const statusText = ref('Waiting for initialization...')
let timer: number | undefined

const mode = useColorMode()
const lastClickPosition = ref<{ clientX: number; clientY: number } | undefined>(undefined)

const configOptions = computed(() => {
    if (!files.value) return []
    return task.value === 'simulation' ? files.value.simulation_configs : files.value.prediction_configs
})

const currentScenario = computed(() => batch.value?.scenarios[worldIndex.value] ?? null)
const selectedAgentValue = computed({
    get: () => (selectedAgentId.value === null ? 'all' : String(selectedAgentId.value)),
    set: (value: string) => {
        selectedAgentId.value = value === 'all' ? null : Number(value)
    }
})
const sortedAgents = computed<Agent[]>(() => {
    const agents = currentScenario.value?.agents ?? []
    return [...agents].sort((a, b) => {
        const ap = Number(a.target) * 8 + Number(a.controlled) * 4 + Number(a.sdc) * 2 - a.id / 10000
        const bp = Number(b.target) * 8 + Number(b.controlled) * 4 + Number(b.sdc) * 2 - b.id / 10000
        return bp - ap
    })
})
const selectedAgent = computed(() => sortedAgents.value.find((agent) => agent.id === selectedAgentId.value) ?? null)
const maxFrame = computed(() => {
    const rollout = currentScenario.value?.rollout ?? []
    return Math.max(0, ...rollout.map((track) => track.points.length - 1))
})
const selectedTrackCounts = computed(() => {
    const scenario = currentScenario.value
    if (!scenario) return { future: 0, predictions: 0, rollout: 0 }
    const agentId = selectedAgentId.value
    return {
        future: agentId === null ? scenario.agents.filter((agent) => agent.future.length > 0).length : selectedAgent.value?.future.length ? 1 : 0,
        predictions: scenario.predictions?.filter((track) => agentId === null || track.agent_id === agentId).length ?? 0,
        rollout: scenario.rollout?.filter((track) => agentId === null || track.agent_id === agentId).length ?? 0
    }
})
const selectedRolloutTrack = computed(() => {
    const agentId = selectedAgentId.value
    if (agentId === null) return null
    return currentScenario.value?.rollout?.find((track) => track.agent_id === agentId) ?? null
})
const metadata = computed(() => {
    const scenario = currentScenario.value
    return {
        agents: scenario?.agents.length ?? 0,
        map: scenario?.map.length ?? 0,
        predictions: scenario?.predictions?.length ?? 0,
        rollout: scenario?.rollout?.length ?? 0
    }
})
const formatRate = (value: number | undefined) => `${((value ?? 0) * 100).toFixed(1)}%`
const formatSignalValue = (value: number | null) => (value === null ? '-' : value.toFixed(4))

function flagAt(values: number[] | undefined, frameIdx: number, cumulative = false) {
    if (!values?.length) return false
    const idx = Math.max(0, Math.min(frameIdx, values.length - 1))
    return cumulative ? values.slice(0, idx + 1).some(Boolean) : Boolean(values[idx])
}

function trackForAgent(agentId: number): RolloutTrack | undefined {
    return currentScenario.value?.rollout?.find((track) => track.agent_id === agentId)
}

function agentState(agent: Agent): 'collision' | 'offroad' | 'goal' | 'controlled' | 'default' {
    const track = trackForAgent(agent.id)
    if (flagAt(track?.collision, frame.value)) return 'collision'
    if (flagAt(track?.offroad, frame.value)) return 'offroad'
    if (flagAt(track?.goal_reached, frame.value, true)) return 'goal'
    if (track?.controlled) return 'controlled'
    return 'default'
}

function agentStateColor(agent: Agent) {
    const state = agentState(agent)
    if (state === 'collision') return '#ef4444'
    if (state === 'offroad') return '#f97316'
    if (state === 'goal') return '#22c55e'
    if (state === 'controlled') return '#3b82f6'
    return '#64748b'
}

function cleanSeries(values: Array<number | null> | undefined) {
    const series = values?.map((value) => (typeof value === 'number' && Number.isFinite(value) ? value : null)) ?? []
    return series.some((value) => value !== null) ? series : null
}

function buildSeriesChartData(track: RolloutTrack | null, key: 'reward' | 'value', label: string, color: string): ChartData<'line'> | null {
    const series = cleanSeries(track?.[key])
    if (!series) return null
    return {
        labels: series.map((_, index) => String(index)),
        datasets: [
            {
                label,
                data: series,
                borderColor: color,
                backgroundColor: color,
                fill: false,
                tension: 0.15,
                pointRadius: 0,
                borderWidth: 2
            }
        ]
    }
}

function seriesValueAt(values: Array<number | null> | undefined) {
    if (!values?.length) return null
    const idx = Math.max(0, Math.min(frame.value, values.length - 1))
    const value = values[idx]
    return typeof value === 'number' && Number.isFinite(value) ? value : null
}

const rewardChartData = computed(() => buildSeriesChartData(selectedRolloutTrack.value, 'reward', 'Reward', '#4A90E2'))
const valueChartData = computed(() => buildSeriesChartData(selectedRolloutTrack.value, 'value', 'Value', '#F5A623'))
const selectedSignalValues = computed(() => ({
    reward: seriesValueAt(selectedRolloutTrack.value?.reward),
    value: seriesValueAt(selectedRolloutTrack.value?.value)
}))

function logStatus(kind: 'INFO' | 'SUCCESS' | 'ERROR', message: string) {
    statusText.value = `${statusText.value}\n[${kind}]     ${message}`.trimStart()
}

function setMode(value: 'light' | 'dark' | 'auto') {
    startThemeTransition(lastClickPosition.value, () => {
        mode.value = value
    })
}

function setTheme(value: unknown) {
    if (value === null || value === undefined) return
    const themeId = String(value)
    startThemeTransition(lastClickPosition.value, () => {
        setStoredThemeId(themeId)
        setThemeOptimized(themeId)
    })
}

async function refreshFiles() {
    files.value = await fetchFiles()
    const nextConfig = configOptions.value[0]
    if (nextConfig && !configOptions.value.includes(configPath.value)) {
        configPath.value = nextConfig
    }
}

async function initialize() {
    isLoading.value = true
    const ckpt = checkpointPath.value === 'None' ? 'none' : checkpointPath.value
    try {
        logStatus('INFO', `Initialize ${task.value} env, config=${configPath.value}, ckpt=${ckpt}`)
        env.value = await initializeEnv({
            task: task.value,
            config_path: configPath.value,
            checkpoint_path: checkpointPath.value === 'None' ? null : checkpointPath.value,
            split: split.value,
            batch_size: batchSize.value
        })
        batchIndex.value = 0
        await loadBatch()
        logStatus('SUCCESS', 'Environment loaded')
        toast.success('Environment loaded')
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        logStatus('ERROR', message)
        toast.error(message)
    } finally {
        isLoading.value = false
    }
}

async function loadBatch() {
    isLoading.value = true
    try {
        logStatus('INFO', `Fetch batch ${batchIndex.value}`)
        batch.value = await fetchBatch(batchIndex.value, batchSize.value)
        worldIndex.value = 0
        frame.value = 0
        scenarioCanvas.value?.resetView()
        logStatus('SUCCESS', `Batch ${batchIndex.value} loaded`)
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        logStatus('ERROR', message)
        toast.error(message)
    } finally {
        isLoading.value = false
    }
}

function stopPlayback() {
    playing.value = false
    if (timer !== undefined) {
        window.clearInterval(timer)
        timer = undefined
    }
}

function togglePlay() {
    playing.value = !playing.value
}

function stepFrame(delta: number) {
    frame.value = Math.min(Math.max(frame.value + delta, 0), maxFrame.value)
}

function resetPlayback() {
    stopPlayback()
    frame.value = 0
}

function navigateWorld(delta: number) {
    const count = batch.value?.scenarios.length ?? 0
    if (!count) return
    worldIndex.value = (worldIndex.value + delta + count) % count
}

function pickDefaultAgent() {
    const agents = sortedAgents.value
    const preferred = agents.find((agent) => agent.target) ?? agents.find((agent) => agent.controlled) ?? agents.find((agent) => agent.sdc) ?? agents[0]
    selectedAgentId.value = preferred?.id ?? null
}

watch(task, () => {
    const nextConfig = configOptions.value[0]
    if (nextConfig && !configOptions.value.includes(configPath.value)) {
        configPath.value = nextConfig
    }
    split.value = task.value === 'simulation' ? 'training' : 'validation'
})

watch(worldIndex, () => {
    frame.value = 0
    pickDefaultAgent()
    scenarioCanvas.value?.resetView()
})

watch(currentScenario, () => {
    pickDefaultAgent()
})

watch(playing, (value) => {
    if (timer !== undefined) {
        window.clearInterval(timer)
        timer = undefined
    }
    if (value) {
        timer = window.setInterval(() => {
            if (maxFrame.value === 0) return
            if (frame.value >= maxFrame.value) {
                stopPlayback()
                return
            }
            frame.value += 1
        }, 1000 / playbackSpeed.value)
    }
})

watch(playbackSpeed, () => {
    if (playing.value) {
        stopPlayback()
        playing.value = true
    }
})

onMounted(async () => {
    window.addEventListener('click', (event) => {
        lastClickPosition.value = { clientX: event.clientX, clientY: event.clientY }
    })
    try {
        await refreshFiles()
        env.value = await fetchEnv()
        task.value = env.value.task
        configPath.value = env.value.config_path
        checkpointPath.value = env.value.checkpoint_path ?? 'None'
        split.value = env.value.split
        batchSize.value = env.value.batch_size
        await loadBatch()
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        logStatus('ERROR', message)
        toast.error(message)
    }
})

onBeforeUnmount(() => {
    stopPlayback()
})
</script>

<template>
    <Toaster rich-colors />
    <div v-auto-animate class="relative h-screen w-screen">
        <ResizablePanelGroup direction="vertical">
            <ResizablePanel class="flex flex-col" :default-size="85">
                <header class="bg-background sticky top-0 z-50 flex w-full items-center justify-between border border-b p-3 px-6 backdrop-blur-xl">
                    <div class="flex items-center gap-4">
                        <h1 class="text-primary flex items-center gap-2 text-xl font-bold">
                            <Icon class="mb-0.5 size-6" icon="lucide:car-front" />
                            Anon Tokyo Visualizer
                        </h1>
                        <div class="text-muted-foreground text-xs">Interactive Prediction and Simulation Visualizer</div>
                    </div>
                    <div class="flex items-center gap-2">
                        <DropdownMenu>
                            <DropdownMenuTrigger as-child>
                                <Button v-auto-animate size="sm" variant="outline">
                                    <Icon class="size-3 scale-100 rotate-4 transition-all dark:scale-0 dark:-rotate-90" icon="lucide:sun" />
                                    <Icon class="absolute size-4 scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" icon="lucide:moon" />
                                    <span class="sr-only">Toggle theme</span>
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                                <DropdownMenuItem @click="setMode('light')">
                                    <Icon class="h-[1.2rem] w-[1.2rem]" icon="lucide:sun" />
                                    Light
                                </DropdownMenuItem>
                                <DropdownMenuItem @click="setMode('dark')">
                                    <Icon class="h-[1.2rem] w-[1.2rem]" icon="lucide:moon" />
                                    Dark
                                </DropdownMenuItem>
                                <DropdownMenuItem @click="setMode('auto')">
                                    <Icon class="h-[1.2rem] w-[1.2rem]" icon="lucide:sun-moon" />
                                    System
                                </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>

                        <Select :model-value="currentThemeId" @update:model-value="setTheme">
                            <SelectTrigger id="theme-selector" class="h-8! w-[160px]" aria-label="Select UI theme">
                                <SelectValue placeholder="Select theme" />
                            </SelectTrigger>
                            <SelectContent align="end">
                                <SelectItem v-for="theme in THEMES" :key="theme.id" :value="theme.id">
                                    {{ theme.label }}
                                </SelectItem>
                            </SelectContent>
                        </Select>

                        <div
                            class="flex items-center gap-1.5 rounded-full px-3 py-2 text-xs font-medium transition-all"
                            :class="
                                env
                                    ? 'bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-300 dark:ring-green-800/60'
                                    : 'bg-red-100 text-red-600 dark:bg-red-900/40 dark:text-red-300 dark:ring-red-800/60'
                            "
                        >
                            <div class="size-2 animate-pulse rounded-full bg-current" />
                            <span>{{ env ? 'Initialized' : 'Not Initialized' }}</span>
                        </div>
                    </div>
                </header>

                <ResizablePanelGroup direction="horizontal">
                    <ResizablePanel :default-size="20">
                        <aside class="scrollbar-thin flex h-full w-full flex-col gap-3 overflow-y-auto p-3">
                            <Card icon="lucide:settings" title="Environment Setup">
                                <div class="grid grid-cols-2 gap-3">
                                    <div>
                                        <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Task</Label>
                                        <Select v-model="task">
                                            <SelectTrigger class="w-full">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="prediction">Prediction</SelectItem>
                                                <SelectItem value="simulation">Simulation</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <div>
                                        <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Split</Label>
                                        <Select v-model="split">
                                            <SelectTrigger class="w-full">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem v-for="name in files?.splits ?? ['training', 'validation', 'testing']" :key="name" :value="name">
                                                    {{ name }}
                                                </SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>

                                <div>
                                    <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Config</Label>
                                    <Select v-model="configPath">
                                        <SelectTrigger class="w-full">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem v-for="path in configOptions" :key="path" :value="path">
                                                {{ path }}
                                            </SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>

                                <CheckpointSelector v-model="checkpointPath" :checkpoint-list="files?.checkpoints ?? []" @reload="refreshFiles" />

                                <div>
                                    <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Batch Size</Label>
                                    <Input v-model.number="batchSize" :min="1" type="number" />
                                </div>

                                <template #footer>
                                    <Button class="w-full cursor-pointer px-4 py-2.5 font-medium" :disabled="isLoading" @click="initialize">
                                        <Icon icon="lucide:play" />
                                        Initialize Environment
                                    </Button>
                                </template>
                            </Card>

                            <Card icon="lucide:boxes" title="Batch & World">
                                <div class="grid grid-cols-2 gap-3">
                                    <div>
                                        <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Batch Index</Label>
                                        <Input v-model.number="batchIndex" min="0" type="number" />
                                    </div>
                                    <div>
                                        <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">World Index</Label>
                                        <Input v-model.number="worldIndex" min="0" :max="Math.max((batch?.scenarios.length ?? 1) - 1, 0)" type="number" />
                                    </div>
                                </div>
                                <Button class="w-full cursor-pointer" :disabled="isLoading" variant="outline" @click="loadBatch">
                                    <Icon class="size-4" icon="lucide:refresh-ccw" />
                                    Fetch Batch
                                </Button>
                                <div class="scrollbar-thin max-h-48 space-y-1 overflow-y-auto pr-1">
                                    <Button
                                        v-for="(scenario, index) in batch?.scenarios ?? []"
                                        :key="scenario.id"
                                        class="h-8 w-full justify-start"
                                        :variant="worldIndex === index ? 'default' : 'ghost'"
                                        @click="worldIndex = index"
                                    >
                                        <span class="text-muted-foreground w-8 text-left text-xs">{{ index }}</span>
                                        <span class="truncate">{{ scenario.id }}</span>
                                    </Button>
                                </div>
                            </Card>

                            <Card icon="lucide:circle-play" title="Playback Controls">
                                <div class="flex items-center justify-center gap-2.5">
                                    <Button size="icon-lg" variant="secondary" @click="resetPlayback">
                                        <Icon class="size-4" icon="lucide:undo" />
                                    </Button>
                                    <Button
                                        class="size-10 rounded-full"
                                        :class="playing ? 'bg-red-600 hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600' : ''"
                                        :disabled="maxFrame === 0"
                                        @click="togglePlay"
                                    >
                                        <Icon class="size-5" :icon="playing ? 'lucide:pause' : 'lucide:play'" />
                                    </Button>
                                    <Button size="icon-lg" title="Previous world" variant="secondary" @click="navigateWorld(-1)">
                                        <Icon class="size-4" icon="lucide:arrow-down" />
                                    </Button>
                                    <Button size="icon-lg" variant="secondary" @click="stepFrame(-1)">
                                        <Icon class="size-4" icon="lucide:step-back" />
                                    </Button>
                                    <Button size="icon-lg" variant="secondary" @click="stepFrame(1)">
                                        <Icon class="size-4" icon="lucide:step-forward" />
                                    </Button>
                                    <Button size="icon-lg" title="Next world" variant="secondary" @click="navigateWorld(1)">
                                        <Icon class="size-4" icon="lucide:arrow-up" />
                                    </Button>
                                </div>
                                <div>
                                    <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Playback Speed</Label>
                                    <div class="flex items-center gap-3">
                                        <Slider v-model="playbackSpeeds" :max="10" :min="0.1" :step="0.1" />
                                        <span class="text-muted-foreground min-w-[50px] text-sm font-medium">{{ playbackSpeed.toFixed(1) }}x</span>
                                    </div>
                                </div>
                                <div class="pb-1">
                                    <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Frame</Label>
                                    <div class="flex items-center gap-3">
                                        <Slider v-model="frameSlider" :max="Math.max(maxFrame, 0)" :min="0" :step="1" />
                                        <span class="text-muted-foreground min-w-[50px] shrink-0 text-sm font-medium">{{ `${frame} / ${maxFrame}` }}</span>
                                    </div>
                                </div>
                            </Card>

                            <Card icon="lucide:eye" title="View Controls">
                                <div class="flex items-center gap-2">
                                    <Icon class="size-5" icon="lucide:bus-front" />
                                    <Label class="block font-medium">Show Road Graph</Label>
                                    <Switch v-model="showMap" class="ml-auto" />
                                </div>
                                <div class="flex items-center gap-2">
                                    <Icon class="size-5" icon="lucide:route" />
                                    <Label class="block font-medium">Show Ground Truth</Label>
                                    <Switch v-model="showGroundTruth" class="ml-auto" />
                                </div>
                                <div class="flex items-center gap-2">
                                    <Icon class="size-5" icon="lucide:brain" />
                                    <Label class="block font-medium">Show Prediction / Rollout</Label>
                                    <Switch v-model="showPredictions" class="ml-auto" />
                                </div>
                                <div class="grid grid-cols-3 gap-3">
                                    <Button class="h-auto cursor-pointer flex-col gap-1" variant="secondary" @click="scenarioCanvas?.zoomIn">
                                        <Icon class="size-5" icon="lucide:zoom-in" />
                                        Zoom In
                                    </Button>
                                    <Button class="h-auto cursor-pointer flex-col gap-1" variant="secondary" @click="scenarioCanvas?.zoomOut">
                                        <Icon class="size-5" icon="lucide:zoom-out" />
                                        Zoom Out
                                    </Button>
                                    <Button class="h-auto cursor-pointer flex-col gap-1" variant="secondary" @click="scenarioCanvas?.resetView">
                                        <Icon class="size-5" icon="lucide:scan" />
                                        Reset
                                    </Button>
                                </div>
                            </Card>
                        </aside>
                    </ResizablePanel>

                    <ResizableHandle with-handle />

                    <ResizablePanel class="relative flex items-center justify-center overflow-hidden backdrop-blur-xl" :default-size="60">
                        <ScenarioCanvas
                            ref="scenarioCanvas"
                            :frame="frame"
                            :scenario="currentScenario"
                            :selected-agent-id="selectedAgentId"
                            :show-ground-truth="showGroundTruth"
                            :show-map="showMap"
                            :show-predictions="showPredictions"
                        />
                    </ResizablePanel>

                    <ResizableHandle with-handle />

                    <ResizablePanel :default-size="20">
                        <aside class="scrollbar-thin flex h-full w-full flex-col gap-3 overflow-y-auto p-3">
                            <Card icon="lucide:chart-no-axes-combined" title="Statistics">
                                <div class="space-y-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-muted-foreground">Dataset</span>
                                        <span class="font-medium">{{ env?.dataset_size ?? '-' }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-muted-foreground">Agents</span>
                                        <span class="font-medium">{{ metadata.agents }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-muted-foreground">Map Elements</span>
                                        <span class="font-medium">{{ metadata.map }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-muted-foreground">Predictions</span>
                                        <span class="font-medium">{{ metadata.predictions }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-muted-foreground">Rollouts</span>
                                        <span class="font-medium">{{ metadata.rollout }}</span>
                                    </div>
                                </div>
                            </Card>

                            <Card v-if="batch?.metrics || currentScenario?.metrics" icon="lucide:activity" title="Rollout Metrics">
                                <div class="space-y-3 text-sm">
                                    <div v-if="batch?.metrics" class="space-y-1.5">
                                        <div class="text-muted-foreground text-xs font-semibold tracking-wide uppercase">Batch</div>
                                        <div class="grid grid-cols-2 gap-2">
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Collision</div>
                                                <div class="font-semibold">{{ formatRate(batch.metrics.collision_rate) }}</div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Offroad</div>
                                                <div class="font-semibold">{{ formatRate(batch.metrics.offroad_rate) }}</div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Goal</div>
                                                <div class="font-semibold">{{ formatRate(batch.metrics.goal_reaching_rate) }}</div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Done</div>
                                                <div class="font-semibold">{{ formatRate(batch.metrics.done_rate) }}</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div v-if="currentScenario?.metrics" class="space-y-1.5">
                                        <div class="text-muted-foreground text-xs font-semibold tracking-wide uppercase">World</div>
                                        <div class="grid grid-cols-2 gap-2">
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Collision</div>
                                                <div class="font-semibold">
                                                    {{ formatRate(currentScenario.metrics.collision_rate) }}
                                                    <span class="text-muted-foreground text-xs">({{ currentScenario.metrics.collision_count }}/{{ currentScenario.metrics.controlled_count }})</span>
                                                </div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Offroad</div>
                                                <div class="font-semibold">
                                                    {{ formatRate(currentScenario.metrics.offroad_rate) }}
                                                    <span class="text-muted-foreground text-xs">({{ currentScenario.metrics.offroad_count }}/{{ currentScenario.metrics.controlled_count }})</span>
                                                </div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Goal</div>
                                                <div class="font-semibold">
                                                    {{ formatRate(currentScenario.metrics.goal_reaching_rate) }}
                                                    <span class="text-muted-foreground text-xs">({{ currentScenario.metrics.goal_reached_count }}/{{ currentScenario.metrics.controlled_count }})</span>
                                                </div>
                                            </div>
                                            <div class="rounded-md border p-2">
                                                <div class="text-muted-foreground text-xs">Done</div>
                                                <div class="font-semibold">
                                                    {{ formatRate(currentScenario.metrics.done_rate) }}
                                                    <span class="text-muted-foreground text-xs">({{ currentScenario.metrics.done_count }}/{{ currentScenario.metrics.controlled_count }})</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </Card>

                            <Card icon="lucide:mouse-pointer-2" title="Agent Selection">
                                <div>
                                    <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">Visible Trajectories</Label>
                                    <Select v-model="selectedAgentValue">
                                        <SelectTrigger class="w-full">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="all">All agents</SelectItem>
                                            <SelectItem v-for="agent in sortedAgents" :key="agent.id" :value="String(agent.id)">
                                                #{{ agent.id }} {{ agent.type }}{{ agent.target ? ' target' : agent.controlled ? ' controlled' : agent.sdc ? ' sdc' : '' }}
                                            </SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div class="grid grid-cols-3 gap-2 text-center text-xs">
                                    <div class="rounded-md border p-2">
                                        <div class="text-muted-foreground">GT</div>
                                        <div class="font-semibold">{{ selectedTrackCounts.future }}</div>
                                    </div>
                                    <div class="rounded-md border p-2">
                                        <div class="text-muted-foreground">Pred</div>
                                        <div class="font-semibold">{{ selectedTrackCounts.predictions }}</div>
                                    </div>
                                    <div class="rounded-md border p-2">
                                        <div class="text-muted-foreground">Rollout</div>
                                        <div class="font-semibold">{{ selectedTrackCounts.rollout }}</div>
                                    </div>
                                </div>
                                <div class="scrollbar-thin max-h-64 space-y-1 overflow-y-auto pr-1">
                                    <Button
                                        v-for="agent in sortedAgents"
                                        :key="agent.id"
                                        class="h-9 w-full justify-start gap-2"
                                        :variant="selectedAgentId === agent.id ? 'default' : 'ghost'"
                                        @click="selectedAgentId = agent.id"
                                    >
                                        <span
                                            class="size-2.5 shrink-0 rounded-full ring-1 ring-black/15 ring-inset"
                                            :style="{ backgroundColor: agentStateColor(agent) }"
                                        />
                                        <span class="w-8 text-left text-xs tabular-nums">#{{ agent.id }}</span>
                                        <span class="truncate">{{ agent.type }}</span>
                                        <Badge v-if="agent.target" class="ml-auto" variant="secondary">target</Badge>
                                        <Badge v-else-if="agent.controlled" class="ml-auto" variant="secondary">ctrl</Badge>
                                        <Badge v-else-if="agent.sdc" class="ml-auto" variant="secondary">sdc</Badge>
                                    </Button>
                                </div>
                            </Card>

                            <Card icon="lucide:chart-line" title="Agent Signals">
                                <div class="flex items-center justify-between gap-2 text-sm">
                                    <div class="min-w-0">
                                        <div class="font-medium">
                                            {{ selectedAgent ? `Agent #${selectedAgent.id}` : 'All agents' }}
                                        </div>
                                        <div class="text-muted-foreground text-xs">
                                            {{ selectedRolloutTrack?.controlled ? 'controlled rollout' : 'rollout series' }}
                                        </div>
                                    </div>
                                    <div class="grid shrink-0 grid-cols-2 gap-2 text-xs">
                                        <div class="rounded-md border px-2 py-1">
                                            <div class="text-muted-foreground">Reward</div>
                                            <div class="font-mono font-semibold">{{ formatSignalValue(selectedSignalValues.reward) }}</div>
                                        </div>
                                        <div class="rounded-md border px-2 py-1">
                                            <div class="text-muted-foreground">Value</div>
                                            <div class="font-mono font-semibold">{{ formatSignalValue(selectedSignalValues.value) }}</div>
                                        </div>
                                    </div>
                                </div>

                                <Tabs v-model="agentSignalTab" class="gap-3">
                                    <TabsList class="grid w-full grid-cols-2">
                                        <TabsTrigger class="cursor-pointer text-xs" value="reward">Reward</TabsTrigger>
                                        <TabsTrigger class="cursor-pointer text-xs" value="value">Value</TabsTrigger>
                                    </TabsList>
                                    <TabsContent class="mt-0" value="reward">
                                        <AgentSeriesChart :chart-data="rewardChartData" empty-text="No reward data" :frame-idx="frame" />
                                    </TabsContent>
                                    <TabsContent class="mt-0" value="value">
                                        <AgentSeriesChart :chart-data="valueChartData" empty-text="No value data" :frame-idx="frame" />
                                    </TabsContent>
                                </Tabs>
                            </Card>

                            <Card icon="lucide:palette" title="Legend">
                                <div class="flex flex-col gap-2">
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#3b82f6] shadow-sm" />
                                        <Label>Controlled Agent</Label>
                                    </div>
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#f59e0b] shadow-sm" />
                                        <Label>SDC</Label>
                                    </div>
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#22c55e] shadow-sm" />
                                        <Label>Target Agent</Label>
                                    </div>
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#64748b] shadow-sm" />
                                        <Label>NPC Agent</Label>
                                    </div>
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#f97316] shadow-sm" />
                                        <Label>Offroad</Label>
                                    </div>
                                    <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                        <div class="h-4 w-4 shrink-0 rounded-full bg-[#ef4444] shadow-sm" />
                                        <Label>Collision</Label>
                                    </div>
                                    <div class="mt-4 border-t pt-3">
                                        <p class="text-muted-foreground mb-1 text-xs font-semibold tracking-wide uppercase">Map</p>
                                        <div class="flex flex-col gap-1">
                                            <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                                <svg class="shrink-0" height="16" viewBox="0 0 52 16" width="52">
                                                    <line stroke="#9370DB" stroke-linecap="round" stroke-width="3" x1="4" x2="48" y1="8" y2="8" />
                                                </svg>
                                                <Label>Curb</Label>
                                            </div>
                                            <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                                <svg class="shrink-0" height="16" viewBox="0 0 52 16" width="52">
                                                    <line class="stroke-foreground" stroke-linecap="round" stroke-width="3" x1="4" x2="48" y1="8" y2="8" />
                                                </svg>
                                                <Label>Solid Line</Label>
                                            </div>
                                            <div class="text-muted-foreground flex items-center gap-2 text-sm">
                                                <svg class="shrink-0" height="16" viewBox="0 0 52 16" width="52">
                                                    <line class="stroke-foreground" stroke-dasharray="6 6" stroke-linecap="round" stroke-width="3" x1="4" x2="48" y1="8" y2="8" />
                                                </svg>
                                                <Label>Dashed Line</Label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        </aside>
                    </ResizablePanel>
                </ResizablePanelGroup>
            </ResizablePanel>

            <ResizableHandle />

            <ResizablePanel>
                <Footer v-model:status-text="statusText" />
            </ResizablePanel>
        </ResizablePanelGroup>

        <div v-if="isLoading" class="absolute inset-0 z-100 flex items-center justify-center bg-black/50">
            <div class="bg-background flex max-w-lg flex-col items-center gap-3 rounded-xl border p-6 shadow">
                <Spinner class="text-primary size-12" />
                <div class="text-muted-foreground wrap-break-words text-center text-sm whitespace-pre-wrap">Loading...</div>
            </div>
        </div>
    </div>
</template>
