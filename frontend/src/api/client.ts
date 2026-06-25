import axios from 'axios'

// In dev: Vite proxies /api → localhost:8000
// In production (Vercel): VITE_API_URL is set to the Railway backend URL
const baseURL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api'

export const apiClient = axios.create({ baseURL })
