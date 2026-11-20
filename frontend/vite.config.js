import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Builds straight into the API's wwwroot so `dotnet run` serves the UI from the same origin.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5080',
      '/health': 'http://localhost:5080',
    },
  },
  build: {
    outDir: '../api/CoinVision.Api/wwwroot',
    emptyOutDir: true,
  },
})
