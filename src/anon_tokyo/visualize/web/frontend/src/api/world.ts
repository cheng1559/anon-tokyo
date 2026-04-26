import { client } from './request'
import type { BatchPayload, EnvInfo, FileCatalog, Scenario } from '@/types/world'

export async function fetchEnv(): Promise<EnvInfo> {
  const response = await client.get<EnvInfo>('/env')
  return response.data
}

export async function fetchFiles(): Promise<FileCatalog> {
  const response = await client.get<FileCatalog>('/files')
  return response.data
}

export async function initializeEnv(payload: {
  task: string
  config_path: string
  checkpoint_path?: string | null
  split?: string | null
  batch_size: number
}): Promise<EnvInfo> {
  const response = await client.post<EnvInfo>('/env', payload)
  return response.data
}

export async function fetchBatch(batchIndex: number, batchSize?: number): Promise<BatchPayload> {
  const response = await client.get<BatchPayload>(`/batch/${batchIndex}`, {
    params: batchSize ? { batch_size: batchSize } : undefined
  })
  return response.data
}

export async function fetchWorld(batchIndex: number, worldIndex: number): Promise<Scenario> {
  const response = await client.get<Scenario>(`/batch/${batchIndex}/world/${worldIndex}`)
  return response.data
}

export async function rolloutWorld(batchIndex: number, worldIndex: number, count: number): Promise<Scenario> {
  const response = await client.post<Scenario>(`/batch/${batchIndex}/world/${worldIndex}/rollout`, { count })
  return response.data
}
