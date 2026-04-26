import { createApp } from 'vue'
import { autoAnimatePlugin } from '@formkit/auto-animate/vue'
import '@/assets/style.css'
import App from './App.vue'
import { initTheme } from '@/utils/themeManager'

initTheme()
createApp(App).use(autoAnimatePlugin).mount('#app')
