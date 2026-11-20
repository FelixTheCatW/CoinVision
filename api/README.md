# CoinVision API + UI

Recognition of a coin's year (1830-1837) from photos. The Python project trains and exports the
models; this folder serves them over HTTP and ships a small Vue 3 page for manual use.

## Pieces

| Path | What it is |
|---|---|
| `api/CoinVision.Api` | ASP.NET Core (.NET 10) minimal MVC API, ONNX Runtime, Serilog to console |
| `frontend` | Vue 3 + Vite single-page UI, built into `api/CoinVision.Api/wwwroot` |
| `onnx` | Exported graphs plus `model.json` with the preprocessing metadata (git-ignored) |

## Prerequisites

- .NET 10 SDK
- Node 20+ (only to rebuild the UI)
- The exported models: the API reads `Model:Directory` (default `../../onnx`) and fails fast with a
  clear message when `model.json` is missing.

## Export the models

```bash
uv sync --extra export
uv run coinvision export \
  --checkpoint runs/cv5_combo/fold0/best.pt \
  --checkpoint runs/cv5_combo/fold1/best.pt \
  --checkpoint runs/cv5_combo/fold2/best.pt \
  --checkpoint runs/cv5_combo/fold3/best.pt \
  --checkpoint runs/cv5_combo/fold4/best.pt \
  --out-dir onnx
```

Each graph takes `float32 [N, 3, 224, 448]` (RGB, ImageNet-normalised, stretched to 448x224) and
returns raw logits `[N, 4, 10]`. The export verifies every graph against torch via onnxruntime and
records `max_abs_diff` in `model.json` (measured: 6e-06).

## Run

```bash
dotnet run --project api/CoinVision.Api            # http://localhost:5080
```

The UI is served from the same origin at `/`. To rebuild it:

```bash
cd frontend && npm install && npm run build        # writes into api/CoinVision.Api/wwwroot
npm run dev                                        # or: dev server on :5173 with a proxy to :5080
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/predict` | multipart/form-data, 1..50 files, field name ignored |
| POST | `/api/predict/single` | raw request body, optional `X-Filename` header |
| GET | `/api/info` | model identity, limits, contract summary |
| GET | `/health` | liveness + model summary |
| GET | `/openapi/v1.json` | OpenAPI document |
| GET | `/scalar/v1` | interactive reference |

### Examples

```bash
curl -X POST http://localhost:5080/api/predict \
  -F "files=@coin1.jpg" -F "files=@coin2.webp" -F "files=@coin3.png"

curl -X POST http://localhost:5080/api/predict/single \
  -H "Content-Type: image/jpeg" -H "X-Filename: coin.jpg" --data-binary @coin.jpg

curl -s http://localhost:5080/api/info
```

### Response (abridged)

```json
{
  "requestId": "00-9d28...-02",
  "receivedAt": "2026-11-20T18:02:11.428+00:00",
  "model": { "count": 5, "inputWidth": 448, "inputHeight": 224, "digits": 4, "classes": 10,
             "files": ["coinvision_0.onnx", "..."], "sources": ["runs/cv5_combo/fold0/best.pt", "..."] },
  "items": [
    {
      "fileName": "coin1.jpg", "contentType": "image/jpeg", "sizeBytes": 99445,
      "originalWidth": 994, "originalHeight": 496, "ok": true, "error": null,
      "year": "1830", "confidence": 0.70845,
      "digits": [ { "position": 0, "digit": 1, "confidence": 0.992 }, "..." ],
      "alternatives": [ { "year": "1832", "probability": 0.159 }, "..." ],
      "timingMs": { "decodeMs": 17.6, "preprocessMs": 28.6, "inferenceMs": 223.1, "totalMs": 269.3 },
      "sha256": "19803a0f2498..."
    }
  ],
  "summary": {
    "count": 3, "failed": 0, "wallMs": 968.0,
    "totalMs": { "mean": 307.0, "median": 303.6, "p95": 330.3, "max": 330.3, "total": 921.0 },
    "inferenceMs": { "mean": 267.0, "median": 267.0, "p95": 267.0, "max": 267.0, "total": 801.0 },
    "confidenceMean": 0.654, "confidenceMin": 0.561,
    "confidenceHistogram": { "<0.50": 0, "0.50-0.70": 2, "0.70-0.90": 1, "0.90-0.99": 0, "0.99-1.00": 0 },
    "years": { "1830": 2, "1833": 1 }
  },
  "warnings": []
}
```

## Semantics worth knowing

- **Ensemble.** All graphs in `onnx/model.json` are averaged (mean softmax); that is the multi-seed
  bag measured in `EXPERIMENTS.md`, which mostly helps the rare 1830 class.
- **confidence** is the product of the four digit probabilities, i.e. the probability of the whole
  year. `digits[3].confidence` isolates the last digit, which is what actually distinguishes these
  years. Both are reported.
- **alternatives** are the runner-up years for the predicted prefix; they show how close the call was.
- **Timing.** `decodeMs` and `preprocessMs` are per image; the inference runs as one batched pass and
  its share is attributed per image, so `sum(items.inferenceMs) ≈ summary.inferenceMs.total`.
- **Per-file failures** never abort the batch: the item comes back with `ok: false` and an `error`,
  and the rest are still recognised. Oversized files are reported in `warnings`.
- **Logging.** Serilog writes one line per request and per image (year, confidence, timings) to the
  console; `UseSerilogRequestLogging` adds one line per HTTP call.
- **Limits** live in `appsettings.json` (`Uploads:MaxFilesPerRequest`, `Uploads:MaxFileBytes`) and are
  echoed by `/api/info`, so a client can discover them instead of hard-coding.

## Verified

- ONNX export vs torch: max |difference| 6e-06.
- API vs Python on 8 images: 8/8 identical years, largest confidence delta 0.021 (the difference
  between Skia's resampler and torchvision's antialiased bilinear).
- UI: file picker, drag & drop and Ctrl+V all enqueue images; each result renders as its own card
  with the year, confidence, per-digit bars, alternatives, timings and hashes.
