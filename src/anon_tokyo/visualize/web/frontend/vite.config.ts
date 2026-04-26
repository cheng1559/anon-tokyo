import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'
import tsconfigPaths from 'vite-tsconfig-paths'

export default defineConfig({
    base: '/absproxy/5173/',
    plugins: [vue(), tsconfigPaths(), tailwindcss()],
    server: {
        host: '0.0.0.0', // Change this to a valid IP address if needed
        port: 5173, // Optional otherwise your app will start on default port
        watch: {
            usePolling: true,
            interval: 1000
        }
    },
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        }
    }
})
