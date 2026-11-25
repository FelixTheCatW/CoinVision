import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// The API serves the built page from its own wwwroot, so one process serves both.
export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: '../api/CoinVision.DateApi/wwwroot',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5199',
    },
  },
})
