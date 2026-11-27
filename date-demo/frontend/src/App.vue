<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const info = ref(null)
const file = ref(null)
const localUrl = ref(null)
const result = ref(null)
const loading = ref(false)
const error = ref(null)
const dragging = ref(false)
const input = ref(null)

const verdictClass = computed(() => (result.value?.hasDate ? 'yes' : 'no'))
const verdictText = computed(() => (result.value?.hasDate ? 'Дата есть' : 'Даты нет'))

const stages = computed(() => {
  const t = result.value?.timingMs
  if (!t) return []
  return [
    ['Декодирование', t.decode],
    ['Поиск монет', t.detect],
    ['Разметка окон', t.sweep],
    ['Кропы окон', t.crop],
    ['Скорер', t.score],
    ['Чтение года', t.read],
    ['Итого', t.total],
  ]
})

const stageMax = computed(() => Math.max(1, ...stages.value.map(([, value]) => value)))

function pick() {
  input.value?.click()
}

function onFile(event) {
  const chosen = event.target.files?.[0]
  if (chosen) send(chosen)
}

function onDrop(event) {
  dragging.value = false
  const chosen = event.dataTransfer?.files?.[0]
  if (chosen) send(chosen)
}

// Ctrl+V: картинки из буфера приходят как DataTransferItem вида 'file'.
function onPaste(event) {
  const items = event.clipboardData?.items
  if (!items) return
  for (const item of items) {
    if (item.kind !== 'file' || !item.type.startsWith('image/')) continue
    const blob = item.getAsFile()
    if (!blob) continue
    event.preventDefault()
    const extension = item.type === 'image/jpeg' ? 'jpg' : item.type.split('/')[1] || 'png'
    send(new File([blob], `clipboard.${extension}`, { type: item.type }))
    return
  }
}

async function send(chosen) {
  file.value = chosen
  error.value = null
  result.value = null
  if (localUrl.value) URL.revokeObjectURL(localUrl.value)
  localUrl.value = URL.createObjectURL(chosen)
  loading.value = true
  try {
    const body = new FormData()
    body.append('file', chosen, chosen.name || 'clipboard.png')
    const response = await fetch('/api/detect', { method: 'POST', body })
    if (!response.ok) {
      const problem = await response.json().catch(() => null)
      throw new Error(problem?.detail ?? `HTTP ${response.status}`)
    }
    const payload = await response.json()
    if (!payload.ok) throw new Error(payload.error ?? 'не удалось обработать изображение')
    result.value = payload
  } catch (exception) {
    error.value = String(exception.message ?? exception)
  } finally {
    loading.value = false
  }
}

function cropSrc(payload) {
  return payload?.cropPng ? `data:image/png;base64,${payload.cropPng}` : null
}

function readerSrc(payload) {
  return payload?.readerCropPng ? `data:image/png;base64,${payload.readerCropPng}` : null
}

function previewSrc(payload) {
  return payload?.previewPng ? `data:image/png;base64,${payload.previewPng}` : null
}

onMounted(async () => {
  window.addEventListener('paste', onPaste)
  try {
    const response = await fetch('/api/info')
    if (response.ok) info.value = await response.json()
  } catch {
    info.value = null
  }
})

onUnmounted(() => {
  window.removeEventListener('paste', onPaste)
})
</script>

<template>
  <div class="wrap">
    <header>
      <h1>CoinVision — есть ли на монете дата</h1>
      <p>
        Загрузите фотографию монеты. Сервис находит круги монет, размечает кольцо на окна,
        скорер выбирает окно, в котором год помещается целиком, и отдаёт его кроп.
        Дальше читатель, обученный на кропах детектора, называет четыре цифры года —
        вместе с уверенностью и тем кропом, который он видел.
      </p>
    </header>

    <div
      class="drop"
      :class="{ hot: dragging }"
      @click="pick"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
    >
      <strong>{{ loading ? 'Считаю…' : 'Перетащите фото монеты, вставьте из буфера (Ctrl+V) или нажмите, чтобы выбрать' }}</strong>
      <span>PNG, JPEG, WebP · до 25 МБ</span>
      <input ref="input" type="file" accept="image/*" hidden @change="onFile" />
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div v-if="info" class="chips">
      <span class="chip mono">{{ info.model }}</span>
      <span class="chip mono">кроп {{ info.cropWidth }}×{{ info.cropHeight }}</span>
      <span class="chip mono">дуга {{ info.arcDegrees }}° / шаг {{ info.stepDegrees }}°</span>
      <span class="chip mono">полосы {{ info.bands.map((b) => b.join('–')).join(', ') }}</span>
      <span class="chip mono">порог {{ info.threshold }}</span>
    </div>

    <div v-if="result" class="grid">
      <section class="panel">
        <h2>Фото и найденное окно</h2>
        <div class="preview">
          <img :src="previewSrc(result)" alt="фото с найденными монетами и выбранным окном" />
        </div>
        <div class="chips">
          <span class="chip">монет: {{ result.coins.length }}</span>
          <span class="chip">окон: {{ result.windowCount }}</span>
          <span class="chip">файл: {{ result.fileName }}</span>
          <span class="chip">{{ result.width }}×{{ result.height }}</span>
        </div>
      </section>

      <section class="panel">
        <h2>Год на монете</h2>
        <div v-if="result.year" class="year">
          <div>
            <div class="digits">{{ result.year }}</div>
            <div class="sub">
              уверенность {{ (result.yearConfidence * 100).toFixed(0) }}% ·
              {{ result.readerFromDetector ? 'квад детектора' : 'квада нет' }}
            </div>
            <div class="digitbars">
              <i v-for="(p, index) in result.yearDigits" :key="index" :title="p.toFixed(3)">
                <b :style="{ width: Math.round(p * 100) + '%' }"></b>
              </i>
            </div>
          </div>
          <img v-if="result.readerCropPng" :src="readerSrc(result)" alt="что прочитал читатель" />
        </div>
        <p v-else class="muted">
          детектор не предложил квад — читать нечего, это фото остаётся за кадровым читателем
        </p>

        <h2>Вердикт</h2>
        <div class="verdict" :class="verdictClass">
          <span class="dot"></span>
          <div>
            <div class="title">{{ verdictText }}</div>
            <div class="sub">
              p = {{ result.bestScore.toFixed(3) }} при пороге {{ result.threshold }}
            </div>
          </div>
        </div>

        <h2>Вырезанная область</h2>
        <div class="crop">
          <img v-if="result.cropPng" :src="cropSrc(result)" alt="вырезанная область" />
          <p v-else class="muted">окно не найдено — монета не детектирована</p>
        </div>
      </section>

      <section class="panel">
        <h2>Технические показатели</h2>
        <table class="metrics">
          <tr v-for="[label, value] in stages" :key="label">
            <th>{{ label }}</th>
            <td class="num">{{ value.toFixed(1) }} мс</td>
            <td style="width: 34%">
              <div class="bar"><i :style="{ width: Math.round((value / stageMax) * 100) + '%' }"></i></div>
            </td>
          </tr>
          <tr>
            <th>Размер файла</th>
            <td class="num" colspan="2">{{ (result.sizeBytes / 1024).toFixed(0) }} КБ</td>
          </tr>
          <tr v-for="(coin, index) in result.coins" :key="'c' + index">
            <th>Монета {{ index + 1 }}</th>
            <td class="num" colspan="2">
              центр ({{ coin.cx }}, {{ coin.cy }}), r = {{ coin.radius }}
            </td>
          </tr>
        </table>
      </section>

      <section class="panel">
        <h2>Пять лучших окон</h2>
        <table class="windows">
          <thead>
            <tr>
              <th>#</th>
              <th class="right">p</th>
              <th>источник</th>
              <th class="right">размер</th>
              <th class="right">угол</th>
              <th class="right">центр</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="window in result.topWindows" :key="window.index">
              <td class="mono">{{ window.index }}</td>
              <td class="right mono">{{ window.score.toFixed(3) }}</td>
              <td class="mono">{{ window.source === 'detector' ? 'детектор' : 'кольцо' }}</td>
              <td class="right mono">
                {{ window.arcSpan.toFixed(0) }}×{{ window.radialSpan.toFixed(0) }}
              </td>
              <td class="right mono">{{ window.angleDeg.toFixed(0) }}°</td>
              <td class="right mono">
                {{ window.centerX.toFixed(0) }}, {{ window.centerY.toFixed(0) }}
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>

    <p v-if="!result && !loading" class="muted" style="margin-top: 18px">
      Ничего не загружено. Модель одна: скорер окон (5-фолдовый распознаватель года из прошлой
      ветки в этом приложении не используется).
    </p>
  </div>
</template>
