import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,  // 改用 Vite 預設端口，避免與其他服務衝突
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/train-api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/train-api/, '')
      }
    }
  },
  define: {
    'process.env': {
      VITE_API_URL: 'http://localhost:8000',
      VITE_TRAIN_API_URL: 'http://localhost:8001'
    }
  }
})

