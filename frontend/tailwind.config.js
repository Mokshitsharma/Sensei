/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0f172a',
        'bg-surface': '#1e293b',
        'bg-elevated': '#243147',
        'border-color': '#334155',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
        'buy': '#22c55e',
        'sell': '#ef4444',
        'hold': '#f59e0b',
        'accent': '#6366f1',
        'accent-hover': '#4f46e5',
      },
    },
  },
  plugins: [],
}
