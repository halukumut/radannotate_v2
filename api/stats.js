/**
 * Vercel Serverless Function — /api/stats
 *
 * GET  /api/stats              → tüm oturumları + aggregate döner
 * GET  /api/stats?mode=solo    → filtreli
 * GET  /api/stats?summary=true → ham bbox'lar gizli
 * POST /api/stats              → yeni oturum kaydeder
 * PATCH /api/stats             → oturum içinde feedback günceller (body: {imageId, feedback})
 *
 * Ortam değişkenleri (Vercel Dashboard > Settings > Environment Variables):
 *   UPSTASH_REDIS_REST_URL
 *   UPSTASH_REDIS_REST_TOKEN
 *
 * Upstash ücretsiz Redis: https://console.upstash.com
 * (Yeni database → REST URL + Token kopyala → Vercel'e yapıştır)
 */

import { Redis } from "@upstash/redis";

const SESSIONS_KEY = "radannotate:sessions";

// Redis bağlantısı (env yoksa hata fırlatır — Vercel'de tanımlı olmalı)
  function getRedis() {
    const url   = process.env.UPSTASH_REDIS_REST_URL;
    const token = process.env.UPSTASH_REDIS_REST_TOKEN;
    if (!url || !token) {
      throw new Error(
        "UPSTASH_REDIS_REST_URL ve UPSTASH_REDIS_REST_TOKEN env değişkenleri tanımlı değil."
      );
    }
    return new Redis({ url, token });
}

// ── Yardımcılar ───────────────────────────────────────────────────────────

function avg(arr) {
  if (!arr.length) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function buildSummary(results) {
  const times   = results.map(r => r.time);
  const ious    = results.map(r => r.iou).filter(v => v != null);
  const correct = results.filter(r => r.labelCorrect).length;
  const total   = times.reduce((a, b) => a + b, 0);

  return {
    imageCount:      results.length,
    correctCount:    correct,
    accuracyRate:    results.length ? +(correct / results.length).toFixed(4) : null,
    totalTimeMs:     total,
    avgTimeMs:       times.length   ? +avg(times).toFixed(1)  : null,
    minTimeMs:       times.length   ? Math.min(...times)       : null,
    maxTimeMs:       times.length   ? Math.max(...times)       : null,
    avgIou:          ious.length    ? +avg(ious).toFixed(4)    : null,
    minIou:          ious.length    ? +Math.min(...ious).toFixed(4) : null,
    maxIou:          ious.length    ? +Math.max(...ious).toFixed(4) : null,
    annotatedImages: results.filter(r => r.boxes?.length > 0).length,
    emptyImages:     results.filter(r => !r.boxes?.length).length,
  };
}

function buildAggregate(sessions) {
  if (!sessions.length) return null;
  const allTimes   = sessions.flatMap(s => (s.results || []).map(r => r.time));
  const allIous    = sessions.flatMap(s => (s.results || []).map(r => r.iou).filter(Boolean));
  const allCorrect = sessions.reduce((acc, s) => acc + (s.summary?.correctCount || 0), 0);
  const allImages  = sessions.reduce((acc, s) => acc + (s.summary?.imageCount  || 0), 0);

  return {
    totalSessions:      sessions.length,
    totalImages:        allImages,
    overallAccuracy:    allImages    ? +(allCorrect / allImages).toFixed(4) : null,
    overallTotalTimeMs: allTimes.reduce((a, b) => a + b, 0),
    overallAvgTimeMs:   allTimes.length ? +avg(allTimes).toFixed(1)  : null,
    overallAvgIou:      allIous.length  ? +avg(allIous).toFixed(4)   : null,
    byMode: {
      solo: sessions.filter(s => s.mode === "solo").length,
      ai:   sessions.filter(s => s.mode === "ai").length,
    },
  };
}

// ── Handler ───────────────────────────────────────────────────────────────

export default async function handler(req, res) {
  // CORS
  res.setHeader("Access-Control-Allow-Origin",  "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();

  let redis;
  try {
    redis = getRedis();
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }

  // ── POST /api/stats ───────────────────────────────────────────────────
  if (req.method === "POST") {
    const { sessionId, mode, physicianId, completedAt, results } = req.body || {};

    if (!Array.isArray(results)) {
      return res.status(400).json({ ok: false, error: "results dizisi zorunlu" });
    }

    const summary = buildSummary(results);
    const record  = {
      sessionId:   sessionId   || `session_${Date.now()}`,
      mode:        mode        || "unknown",
      physicianId: physicianId || "anonymous",
      completedAt: completedAt || new Date().toISOString(),
      summary,
      results,
    };

    // Redis listesine ekle (sağdan push — en yeni sonda)
    await redis.rpush(SESSIONS_KEY, JSON.stringify(record));

    return res.status(201).json({ ok: true, sessionId: record.sessionId, summary });
  }

  // ── GET /api/stats ────────────────────────────────────────────────────
  if (req.method === "GET") {
    const raw = await redis.lrange(SESSIONS_KEY, 0, -1);

    // Redis lrange string dizisi döner — parse et
    let sessions = raw.map(item =>
      typeof item === "string" ? JSON.parse(item) : item
    );

    // Filtreler
    const { mode, physician, summary: summaryOnly } = req.query;
    if (mode)      sessions = sessions.filter(s => s.mode        === mode);
    if (physician) sessions = sessions.filter(s => s.physicianId === physician);

    const output = summaryOnly === "true"
      ? sessions.map(({ results: _r, ...rest }) => rest)
      : sessions;

    return res.status(200).json({
      totalSessions: sessions.length,
      aggregate:     buildAggregate(sessions),
      sessions:      output,
    });
  }

  // ── PATCH /api/stats ──────────────────────────────────────────────────
  // Feedback güncelle (oturum içindeki belirli bir görsel için)
  if (req.method === "PATCH") {
    const { imageId, feedback } = req.body || {};

    if (!imageId) {
      return res.status(400).json({ ok: false, error: "imageId gerekli" });
    }

    // Tüm oturumları al ve ara
    const raw = await redis.lrange(SESSIONS_KEY, 0, -1);
    let sessions = raw.map(item =>
      typeof item === "string" ? JSON.parse(item) : item
    );

    let updated = false;

    // Son oturumda feedback'i güncelle (genellikle en son gönderilen oturum)
    if (sessions.length > 0) {
      const lastSession = sessions[sessions.length - 1];
      if (lastSession.results) {
        const resultIdx = lastSession.results.findIndex(r => r.imageId === imageId);
        if (resultIdx >= 0) {
          lastSession.results[resultIdx].feedback = feedback;
          // Tüm listeyi güncelle
          await redis.del(SESSIONS_KEY);
          for (const session of sessions) {
            await redis.rpush(SESSIONS_KEY, JSON.stringify(session));
          }
          updated = true;
        }
      }
    }

    return res.status(updated ? 200 : 404).json({
      ok: updated,
      message: updated ? "Feedback kaydedildi" : "Görsel bulunamadı",
    });
  }

  return res.status(405).json({ ok: false, error: "Method not allowed" });
}
