<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const cards = ref([])
const summary = ref(null)
const model = ref(null)
const warnings = ref([])
const dragging = ref(false)
const sending = ref(false)
const error = ref(null)
const input = ref(null)
let requestCounter = 0

const pending = computed(() => cards.value.filter((card) => card.state === 'pending').length)
const accepted = computed(() => cards.value.filter((card) => card.state === 'done' && card.item.ok))
const failed = computed(() => cards.value.filter((card) => card.state === 'done' && !card.item.ok))

const apiBase = import.meta.env.VITE_API_BASE ?? ''

function fmt(value, digits = 1) {
  return typeof value === 'number' ? value.toFixed(digits) : '—'
}

function fmtBytes(bytes) {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`
}

function confidenceClass(confidence) {
  if (confidence === null || confidence === undefined) return 'unknown'
  if (confidence >= 0.9) return 'high'
  if (confidence >= 0.7) return 'medium'
  return 'low'
}

function makeCard(file) {
  const id = ++requestCounter
  return {
    id,
    name: file.name,
    size: file.size,
    preview: URL.createObjectURL(file),
    state: 'pending',
    item: null,
  }
}

async function upload(files) {
  const list = Array.from(files ?? []).filter((file) => file && file.size >= 0)
  if (list.length === 0) return

  error.value = null
  // Keep the batches small so each response fills its own cards as it lands.
  const batchSize = 12
  for (let start = 0; start < list.length; start += batchSize) {
    const batch = list.slice(start, start + batchSize)
    const batchCards = batch.map(makeCard)
    cards.value = [...cards.value, ...batchCards]
    await sendBatch(batch, batchCards)
  }
}

async function sendBatch(files, batchCards) {
  sending.value = true
  const form = new FormData()
  files.forEach((file) => form.append('files', file, file.name))

  try {
    const response = await fetch(`${apiBase}/api/predict`, { method: 'POST', body: form })
    if (!response.ok) {
      const problem = await response.json().catch(() => null)
      throw new Error(problem?.detail ?? problem?.title ?? `HTTP ${response.status}`)
    }
    const payload = await response.json()
    model.value = payload.model
    summary.value = payload.summary
    warnings.value = payload.warnings ?? []

    batchCards.forEach((card, index) => {
      const item = payload.items[index]
      card.item = item ?? { ok: false, error: 'no result returned', fileName: card.name }
      card.state = 'done'
    })
  } catch (exception) {
    error.value = String(exception.message ?? exception)
    batchCards.forEach((card) => {
      card.item = { ok: false, error: error.value, fileName: card.name }
      card.state = 'done'
    })
  } finally {
    sending.value = false
  }
}

function onPick(event) {
  const files = event.target.files
  upload(files)
  event.target.value = ''
}

function onDrop(event) {
  event.preventDefault()
  dragging.value = false
  if (event.dataTransfer) upload(event.dataTransfer.files)
}

function onDragOver(event) {
  event.preventDefault()
  dragging.value = true
}

function onDragLeave(event) {
  if (event.currentTarget === event.target) dragging.value = false
}

function onPaste(event) {
  const clipboard = event.clipboardData
  if (!clipboard) return
  const files = []
  for (const item of clipboard.items ?? []) {
    if (item.kind === 'file') {
      const file = item.getAsFile()
      if (file) {
        const stamped = new File([file], file.name || `paste-${Date.now()}.png`, { type: file.type })
        files.push(stamped)
      }
    }
  }
  if (files.length > 0) {
    event.preventDefault()
    upload(files)
  }
}

function clearAll() {
  cards.value.forEach((card) => URL.revokeObjectURL(card.preview))
  cards.value = []
  summary.value = null
  warnings.value = []
  error.value = null
}

async function loadInfo() {
  try {
    const response = await fetch(`${apiBase}/api/info`)
    if (response.ok) {
      const info = await response.json()
      model.value = info.model
    }
  } catch (exception) {
    error.value = `API недоступен: ${exception.message}`
  }
}

onMounted(() => {
  window.addEventListener('paste', onPaste)
  window.addEventListener('dragover', onDragOver)
  window.addEventListener('dragleave', onDragLeave)
  window.addEventListener('drop', onDrop)
  loadInfo()
})

onUnmounted(() => {
  window.removeEventListener('paste', onPaste)
  window.removeEventListener('dragover', onDragOver)
  window.removeEventListener('dragleave', onDragLeave)
  window.removeEventListener('drop', onDrop)
})
</script>

<template>
  <main class="page" :class="{ dragging }">
    <header class="head">
      <div>
        <h1>coinvision</h1>
        <p class="hint">
          Ctrl+V, перетаскивание или выбор файлов. Год определяется по последней цифре (1830–1837).
        </p>
      </div>
      <div class="head-right">
        <div v-if="model" class="model">
          <span>{{ model.count }}×ONNX</span>
          <span>{{ model.inputWidth }}×{{ model.inputHeight }}</span>
          <span>digits {{ model.digits }} / classes {{ model.classes }}</span>
        </div>
        <div class="actions">
          <button type="button" class="primary" @click="input?.click()">Выбрать файлы</button>
          <button type="button" @click="clearAll" :disabled="cards.length === 0">Очистить</button>
        </div>
      </div>
    </header>

    <input ref="input" class="hidden-input" type="file" accept="image/*" multiple @change="onPick" />

    <section v-if="dragging" class="dropveil">Отпустите файлы — они уйдут в API</section>

    <p v-if="error" class="error">Ошибка: {{ error }}</p>
    <p v-if="sending" class="status">Отправка… в очереди {{ pending }}</p>

    <section v-if="summary" class="summary">
      <div class="stat">
        <span class="label">изображений</span>
        <b>{{ summary.count }}</b>
        <span v-if="summary.failed" class="bad">(ошибок {{ summary.failed }})</span>
      </div>
      <div class="stat">
        <span class="label">wall</span>
        <b>{{ fmt(summary.wallMs, 0) }} ms</b>
      </div>
      <div class="stat">
        <span class="label">на изображение</span>
        <b>{{ fmt(summary.totalMs.mean) }} ms</b>
        <span class="muted">med {{ fmt(summary.totalMs.median) }} · p95 {{ fmt(summary.totalMs.p95) }}</span>
      </div>
      <div class="stat">
        <span class="label">инференс</span>
        <b>{{ fmt(summary.inferenceMs.mean) }} ms</b>
        <span class="muted">p95 {{ fmt(summary.inferenceMs.p95) }}</span>
      </div>
      <div class="stat">
        <span class="label">уверенность</span>
        <b>{{ fmt(summary.confidenceMean * 100) }}%</b>
        <span class="muted">min {{ fmt(summary.confidenceMin * 100) }}%</span>
      </div>
      <div class="stat wide">
        <span class="label">годы</span>
        <span class="chips">
          <span v-for="(count, year) in summary.years" :key="year" class="chip">{{ year }}: {{ count }}</span>
        </span>
      </div>
      <div class="stat wide">
        <span class="label">распределение уверенности</span>
        <span class="chips">
          <span v-for="(count, bucket) in summary.confidenceHistogram" :key="bucket" class="chip muted">
            {{ bucket }}: {{ count }}
          </span>
        </span>
      </div>
    </section>

    <ul v-if="warnings.length" class="warnings">
      <li v-for="warning in warnings" :key="warning">{{ warning }}</li>
    </ul>

    <section class="cards">
      <article
        v-for="card in cards"
        :key="card.id"
        class="card"
        :class="[card.state === 'done' ? confidenceClass(card.item?.confidence) : 'pending', { invalid: card.item && !card.item.ok }]"
      >
        <div class="card-media">
          <img :src="card.preview" :alt="card.name" loading="lazy" />
        </div>

        <div class="card-body">
          <div class="card-head">
            <span class="filename" :title="card.name">{{ card.name }}</span>
            <span class="size">{{ fmtBytes(card.size) }}</span>
          </div>

          <p v-if="card.state === 'pending'" class="pending">обрабатывается…</p>

          <template v-else-if="card.item?.ok">
            <div class="year-row">
              <span class="year">{{ card.item.year }}</span>
              <span class="confidence" :class="confidenceClass(card.item.confidence)">
                {{ fmt(card.item.confidence * 100, 1) }}%
              </span>
            </div>

            <div class="digits">
              <div v-for="digit in card.item.digits" :key="digit.position" class="digit">
                <span class="d">{{ digit.digit }}</span>
                <span class="bar"><i :style="{ width: `${digit.confidence * 100}%` }"></i></span>
                <span class="p">{{ fmt(digit.confidence * 100, 0) }}%</span>
              </div>
            </div>

            <div class="alternatives">
              <span class="label">альтернативы</span>
              <span v-for="alternative in card.item.alternatives" :key="alternative.year" class="chip">
                {{ alternative.year }} — {{ fmt(alternative.probability * 100, 1) }}%
              </span>
            </div>

            <dl class="timing">
              <div><dt>decode</dt><dd>{{ fmt(card.item.timingMs.decodeMs) }} ms</dd></div>
              <div><dt>preprocess</dt><dd>{{ fmt(card.item.timingMs.preprocessMs) }} ms</dd></div>
              <div><dt>inference</dt><dd>{{ fmt(card.item.timingMs.inferenceMs) }} ms</dd></div>
              <div><dt>total</dt><dd>{{ fmt(card.item.timingMs.totalMs) }} ms</dd></div>
            </dl>

            <p class="meta">
              {{ card.item.originalWidth }}×{{ card.item.originalHeight }} ·
              {{ fmtBytes(card.item.sizeBytes) }} · sha256 {{ (card.item.sha256 || '').slice(0, 12) }}…
            </p>
          </template>

          <p v-else class="error">не распознано: {{ card.item?.error }}</p>
        </div>
      </article>
    </section>
  </main>
</template>
