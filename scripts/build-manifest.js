#!/usr/bin/env node
/**
 * scripts/build-manifest.js
 *
 * public/data/images/ klasörünü tarar, her görsel için
 * gt/ ve pred/ txt dosyalarını okur, manifest.json üretir.
 *
 * Çalıştır:  node scripts/build-manifest.js
 * Çıktı:     public/data/manifest.json
 *
 * Desteklenen görsel uzantıları: .jpg .jpeg .png .webp
 * GT format:   class cx cy w h          (bir satır = bir bbox)
 * Pred format: class cx cy w h conf     (bir satır = bir bbox)
 */

const fs   = require("fs");
const path = require("path");

const DATA_DIR   = path.join(__dirname, "../public/data");
const IMG_DIR    = path.join(DATA_DIR, "images");
const GT_DIR     = path.join(DATA_DIR, "gt");
const PRED_DIR   = path.join(DATA_DIR, "pred");
const OUT_FILE   = path.join(DATA_DIR, "manifest.json");

const IMG_EXTS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

// ── Yardımcılar ───────────────────────────────────────────────────────────

/** YOLO txt dosyasını okur. Her satır bir bbox döner. */
function parseTxt(filePath, hasCont) {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf8").trim();
  if (!raw) return [];                      // boş dosya → negatif vaka

  return raw.split("\n").map((line, idx) => {
    const parts = line.trim().split(/\s+/).map(Number);
    if (hasCont) {
      // pred: class cx cy w h conf
      const [cls, cx, cy, w, h, conf] = parts;
      return { id: idx, cls, cx, cy, w, h, conf: conf ?? 1.0 };
    } else {
      // gt: class cx cy w h
      const [cls, cx, cy, w, h] = parts;
      return { id: idx, cls, cx, cy, w, h };
    }
  }).filter(b => !isNaN(b.cx));            // hatalı satırları atla
}

// ── Ana akış ─────────────────────────────────────────────────────────────

const imageFiles = fs.readdirSync(IMG_DIR)
  .filter(f => IMG_EXTS.has(path.extname(f).toLowerCase()))
  .sort();                                  // alfabetik sıra

if (!imageFiles.length) {
  console.error("HATA: public/data/images/ içinde görsel bulunamadı.");
  process.exit(1);
}

// ── 25 / 25 bölünme ──────────────────────────────────────────────────────
// İlk yarı (index 0-24)  → Set A: Solo oturum (pred gösterilmez)
// İkinci yarı (index 25-49) → Set B: AI destekli oturum (pred gösterilir)
// Görsel sayısı 50'den farklıysa eşit yarıya bölünür (tek sayıda A bir fazla alır)
const SET_A_COUNT = Math.ceil(imageFiles.length / 2);

const images = imageFiles.map((file, index) => {
  const stem   = path.basename(file, path.extname(file));
  const gtFile   = path.join(GT_DIR,   stem + ".txt");
  const predFile = path.join(PRED_DIR, stem + ".txt");

  // Eksik dosyalar boş dizi döner (hata fırlatmaz)
  const gt   = parseTxt(gtFile,   false);
  const pred = parseTxt(predFile, true);

  const setId = index < SET_A_COUNT ? "A" : "B";

  // Durum etiketi (debug / istatistik için)
  let status;
  if (gt.length > 0 && pred.length > 0) status = "gt_pred";
  else if (gt.length > 0)               status = "gt_only";
  else if (pred.length > 0)             status = "pred_only";
  else                                  status = "negative";

  return {
    id:      index + 1,
    stem,
    file,
    url:     `/data/images/${file}`,
    setId,          // "A" | "B"
    status,         // "gt_pred" | "gt_only" | "pred_only" | "negative"
    gt,             // [ { id, cls, cx, cy, w, h } ]
    pred,           // [ { id, cls, cx, cy, w, h, conf } ]
    hasGt:   gt.length > 0,
    hasPred: pred.length > 0,
  };
});

// ── Set istatistikleri ────────────────────────────────────────────────────
const setA = images.filter(i => i.setId === "A");
const setB = images.filter(i => i.setId === "B");

const manifest = {
  generatedAt: new Date().toISOString(),
  totalImages: images.length,
  setA: { count: setA.length, withGt: setA.filter(i=>i.hasGt).length, withPred: setA.filter(i=>i.hasPred).length, negatives: setA.filter(i=>!i.hasGt&&!i.hasPred).length },
  setB: { count: setB.length, withGt: setB.filter(i=>i.hasGt).length, withPred: setB.filter(i=>i.hasPred).length, negatives: setB.filter(i=>!i.hasGt&&!i.hasPred).length },
  withGt:    images.filter(i => i.hasGt).length,
  withPred:  images.filter(i => i.hasPred).length,
  negatives: images.filter(i => !i.hasGt && !i.hasPred).length,
  statusCounts: {
    gt_pred:  images.filter(i=>i.status==="gt_pred").length,
    gt_only:  images.filter(i=>i.status==="gt_only").length,
    pred_only:images.filter(i=>i.status==="pred_only").length,
    negative: images.filter(i=>i.status==="negative").length,
  },
  images,
};

fs.writeFileSync(OUT_FILE, JSON.stringify(manifest, null, 2), "utf8");

console.log(`✓ manifest.json oluşturuldu → ${OUT_FILE}`);
console.log(`  Toplam görsel : ${manifest.totalImages}`);
console.log(`  Set A (Solo)  : ${manifest.setA.count} görsel  |  GT: ${manifest.setA.withGt}  Pred: ${manifest.setA.withPred}  Negatif: ${manifest.setA.negatives}`);
console.log(`  Set B (AI)    : ${manifest.setB.count} görsel  |  GT: ${manifest.setB.withGt}  Pred: ${manifest.setB.withPred}  Negatif: ${manifest.setB.negatives}`);
console.log(`  Durum dağılımı: gt+pred=${manifest.statusCounts.gt_pred}  gt_only=${manifest.statusCounts.gt_only}  pred_only=${manifest.statusCounts.pred_only}  negatif=${manifest.statusCounts.negative}`);
