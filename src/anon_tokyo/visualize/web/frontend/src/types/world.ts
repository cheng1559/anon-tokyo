export interface Point2 {
  0: number
  1: number
}

export interface MapLine {
  type: number
  points: number[][]
}

export interface Agent {
  id: number
  type: string
  history: number[][]
  future: number[][]
  position: number[]
  size: number[]
  heading: number
  target: boolean
  sdc: boolean
  controlled: boolean
}

export interface PredictionTrack {
  agent_id: number
  mode: number
  score: number
  points: number[][]
}

export interface RolloutTrack {
  agent_id: number
  points: number[][]
  headings?: number[]
  controlled?: boolean
  valid?: number[]
  goal?: number[]
  goal_reached?: number[]
  goal_reached_frame?: number | null
}

export interface Goal {
  agent_id: number
  point: number[]
}

export interface Scenario {
  id: string
  map: MapLine[]
  agents: Agent[]
  predictions?: PredictionTrack[]
  rollout?: RolloutTrack[]
  goals?: Goal[]
}

export interface BatchPayload {
  task: 'prediction' | 'simulation'
  scenarios: Scenario[]
}

export interface EnvInfo {
  task: string
  config_path: string
  checkpoint_path?: string | null
  split: string
  batch_size: number
  dataset_size: number
}

export interface FileCatalog {
  prediction_configs: string[]
  simulation_configs: string[]
  checkpoints: string[]
  splits: string[]
}
