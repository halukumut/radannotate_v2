# RadAnnotate

Radyoloji anotasyon benchmark platformu.  
Hekim solo vs. AI destekli performansını ölçer: bbox IoU, doğruluk, süre.

## Veri yapısı

```
public/data/
├── images/                      ← Görsel dosyalarınız
│   ├── hasta_001_slice_01.jpg   ← hasta_id_slice formatı
│   ├── hasta_001_slice_02.jpg
│   └── ...
├── gt/                          ← Ground truth (YOLO format: class cx cy w h)
│   ├── hasta_001_slice_01.txt   ← Görsel ile aynı isim (uzantı hariç)
│   └── ...                      ← Boş dosya veya eksik = negatif vaka
├── pred/                        ← Model tahminleri (YOLO + conf: class cx cy w h conf)
│   ├── hasta_001_slice_01.txt
│   └── ...
└── manifest.json                ← Otomatik üretilir (npm run manifest)
```

### Görsel bölünmesi (25 / 25)

Görseller alfabetik sıraya göre iki eşit sete ayrılır:

| Set | Oturum | Görseller | Pred gösterilir mi? |
|-----|--------|-----------|---------------------|
| A   | Solo   | 1–25      | Hayır               |
| B   | AI Destekli | 26–50 | Evet (bbox + conf) |

### Desteklenen durumlar

| GT dosyası | Pred dosyası | Durum       | Doğruluk mantığı                  |
|------------|--------------|-------------|-------------------------------------|
| Dolu       | Dolu         | `gt_pred`   | IoU > 0.3 ise doğru                |
| Dolu       | Boş/eksik    | `gt_only`   | IoU > 0.3 ise doğru                |
| Boş/eksik  | Dolu         | `pred_only` | Hekim boş bırakırsa yanlış (FP)    |
| Boş/eksik  | Boş/eksik    | `negative`  | Hekim boş bırakırsa doğru (TN)     |


## Deploy — Vercel (ücretsiz)

### 1. Upstash Redis oluştur

1. [console.upstash.com](https://console.upstash.com) → **Create Database**
2. Region: Frankfurt (Avrupa'ya en yakın)
3. **REST API** sekmesinden şunları kopyala:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`

### 2. Vercel'e deploy et

```bash
# Vercel CLI kur (bir kez)
npm i -g vercel

# Proje dizininde
npm install
vercel deploy

# Env değişkenlerini ekle
vercel env add UPSTASH_REDIS_REST_URL
vercel env add UPSTASH_REDIS_REST_TOKEN

# Production deploy
vercel --prod
```

Alternatif: GitHub'a push → Vercel Dashboard → **New Project** → repo seç → env değişkenlerini UI'dan ekle → Deploy.

### 3. Kontrol et

```
https://your-app.vercel.app/demo       → Uygulama
https://your-app.vercel.app/api/stats  → JSON veriler
```

## API

### POST /api/stats
Oturum sonuçlarını kaydeder.

```json
{
  "sessionId": "session_1234567890",
  "mode": "solo",
  "physicianId": "dr-ayse",
  "completedAt": "2026-05-12T10:00:00Z",
  "results": [
    {
      "imageId": 1,
      "gtLabel": "Pnömoni",
      "label": "2 bbox",
      "labelCorrect": true,
      "boxes": [{ "x": 0.2, "y": 0.3, "w": 0.25, "h": 0.2 }],
      "iou": 0.71,
      "time": 4230
    }
  ]
}
```

Yanıt:
```json
{
  "ok": true,
  "sessionId": "session_1234567890",
  "summary": {
    "imageCount": 10,
    "correctCount": 8,
    "accuracyRate": 0.8,
    "totalTimeMs": 42300,
    "avgTimeMs": 4230.0,
    "minTimeMs": 1800,
    "maxTimeMs": 9100,
    "avgIou": 0.68,
    "annotatedImages": 9,
    "emptyImages": 1
  }
}
```

### GET /api/stats

Tüm oturumları + aggregate özet döner.

| Query param     | Açıklama                              |
|-----------------|---------------------------------------|
| `?mode=solo`    | Yalnızca solo oturumlar               |
| `?mode=ai`      | Yalnızca AI destekli oturumlar        |
| `?physician=id` | Belirli hekim                         |
| `?summary=true` | Ham bbox verilerini yanıta dahil etme |

Yanıt:
```json
{
  "totalSessions": 4,
  "aggregate": {
    "totalSessions": 4,
    "totalImages": 40,
    "overallAccuracy": 0.775,
    "overallTotalTimeMs": 168400,
    "overallAvgTimeMs": 4210.0,
    "overallAvgIou": 0.64,
    "byMode": { "solo": 2, "ai": 2 }
  },
  "sessions": [ ... ]
}
```

## Lokal geliştirme

```bash
# .env.local oluştur (Upstash bilgilerini yapıştır)
cp .env.example .env.local

# Geliştirme sunucusu
npm run dev
# → http://localhost:5173

# Serverless fonksiyonları lokal test için Vercel CLI:
vercel dev
# → http://localhost:3000/api/stats
```
