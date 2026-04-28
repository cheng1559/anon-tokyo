import axios from 'axios'

const DEFAULT_BACKEND_PORT = '8766'

function resolveBaseUrl() {
  if (import.meta.env.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL
  }
  if (typeof window === 'undefined') {
    return `http://localhost:${import.meta.env.VITE_BACKEND_PORT ?? DEFAULT_BACKEND_PORT}`
  }

  const backendPort = import.meta.env.VITE_BACKEND_PORT ?? DEFAULT_BACKEND_PORT
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
  const host = window.location.host
  const segments = window.location.pathname.split('/').filter(Boolean)
  const proxySegment = segments.find((segment) => segment.toLowerCase().endsWith('proxy'))

  if (proxySegment && !host.includes(':')) {
    return `${protocol}//${host}/${proxySegment}/${backendPort}`
  }

  return `${protocol}//${window.location.hostname}:${backendPort}`
}

export const backendUrl = resolveBaseUrl()

export const client = axios.create({
  baseURL: backendUrl,
  timeout: 1000000
})
