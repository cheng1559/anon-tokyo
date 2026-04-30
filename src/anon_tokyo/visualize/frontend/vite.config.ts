import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'
import tsconfigPaths from 'vite-tsconfig-paths'

const DEFAULT_FRONTEND_PORT = 5173

function getCliOption(name: string): string | undefined {
    const optionName = `--${name}`
    const prefixedOptionName = `${optionName}=`

    for (const [index, arg] of process.argv.entries()) {
        if (arg === optionName) {
            return process.argv[index + 1]
        }

        if (arg.startsWith(prefixedOptionName)) {
            return arg.slice(prefixedOptionName.length)
        }
    }

    return undefined
}

function getFrontendPort(): number {
    const rawPort = process.env.VITE_FRONTEND_PORT ?? getCliOption('port')
    const port = Number(rawPort)

    return Number.isInteger(port) && port > 0 ? port : DEFAULT_FRONTEND_PORT
}

const frontendPort = getFrontendPort()

export default defineConfig({
    base: `/absproxy/${frontendPort}/`,
    plugins: [vue(), tsconfigPaths(), tailwindcss()],
    server: {
        host: '0.0.0.0', // Change this to a valid IP address if needed
        port: frontendPort, // Optional otherwise your app will start on default port
        strictPort: true,
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
