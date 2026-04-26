<script lang="ts" setup>
import { computed, ref } from 'vue'
import { Icon } from '@iconify/vue'
import fuzzysort from 'fuzzysort'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

const model = defineModel<string>()

const props = withDefaults(
    defineProps<{
        checkpointList: string[]
        label?: string
    }>(),
    {
        label: 'Checkpoint'
    }
)

const emit = defineEmits<{
    reload: []
}>()

const checkpointSearchOpen = ref(false)
const checkpointQuery = ref('')

const checkpointOptions = computed(() => ['None', ...props.checkpointList])

const checkpointCandidates = computed(() => {
    const all = checkpointOptions.value
    const q = checkpointQuery.value.trim()
    if (!q) return all

    const results = fuzzysort.go(q, all, {
        threshold: -10000,
        limit: 200
    })
    return results.map((r) => r.target)
})

function selectCheckpoint(selectedValue: string) {
    model.value = selectedValue
    checkpointSearchOpen.value = false
}

function reload() {
    emit('reload')
}
</script>

<template>
    <div>
        <Label class="text-muted-foreground mb-1.5 block text-sm font-medium">{{ label }}</Label>
        <div class="flex gap-2">
            <Popover v-model:open="checkpointSearchOpen">
                <PopoverTrigger as-child>
                    <Button
                        class="min-w-0 flex-1 justify-between"
                        :aria-expanded="checkpointSearchOpen"
                        aria-label="Select checkpoint"
                        role="combobox"
                        variant="outline"
                    >
                        <span class="truncate">
                            {{ model || 'Select checkpoint...' }}
                        </span>
                        <Icon class="text-muted-foreground size-4" icon="lucide:chevrons-up-down" />
                    </Button>
                </PopoverTrigger>
                <PopoverContent class="max-h-96 w-auto p-2" align="start">
                    <div class="flex flex-col gap-2">
                        <Input v-model="checkpointQuery" class="h-9" placeholder="Search checkpoint..." />
                        <div class="scrollbar-none max-h-80 overflow-auto">
                            <div v-if="checkpointCandidates.length === 0" class="text-muted-foreground px-2 py-1.5 text-sm">No checkpoint found.</div>
                            <button
                                v-for="checkpoint in checkpointCandidates"
                                :key="checkpoint"
                                class="hover:bg-muted flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-sm"
                                type="button"
                                @click="selectCheckpoint(checkpoint)"
                            >
                                <span class="truncate">{{ checkpoint }}</span>
                                <Icon class="size-4" :class="model === checkpoint ? 'opacity-100' : 'opacity-0'" icon="lucide:check" />
                            </button>
                        </div>
                    </div>
                </PopoverContent>
            </Popover>
            <Button size="icon" variant="outline" @click="reload">
                <Icon class="size-4" icon="lucide:refresh-ccw" />
            </Button>
        </div>
    </div>
</template>
