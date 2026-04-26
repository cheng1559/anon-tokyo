import axios from 'axios'

function resolveBaseUrl() {
  if (import.meta.env.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL
  }
  if (typeof window === 'undefined') {
    return 'http://localhost:8766'
  }

  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
  const host = window.location.host
  const segments = window.location.pathname.split('/').filter(Boolean)
  const proxySegment = segments.find((segment) => segment.toLowerCase().endsWith('proxy'))

  if (proxySegment && !host.includes(':')) {
    return `${protocol}//${host}/${proxySegment}/8766`
  }

  return `${protocol}//${window.location.hostname}:8766`
}

export const backendUrl = resolveBaseUrl()

export const client = axios.create({
  baseURL: backendUrl,
  timeout: 1000000
})
