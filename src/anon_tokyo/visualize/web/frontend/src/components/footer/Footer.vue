<script lang="ts" setup>
import { nextTick, ref, watch } from 'vue'
import { Icon } from '@iconify/vue'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

const statusText = defineModel<string>('status-text')

const textareaRef = ref<typeof Textarea | null>(null)

watch(statusText, async () => {
    await nextTick()
    const el = textareaRef.value?.$el
    if (el) el.scrollTop = el.scrollHeight
})
</script>

<template>
    <footer class="bg-background relative flex h-full items-center justify-between border-t p-2 font-mono">
        <Textarea ref="textareaRef" class="min-h-none scrollbar-thin h-full resize-none" :model-value="statusText" readonly />
        <Button class="absolute top-6 right-8" size="icon-sm" variant="ghost" @click="statusText = ''">
            <Icon class="size-4" icon="lucide:trash" />
        </Button>
    </footer>
</template>
