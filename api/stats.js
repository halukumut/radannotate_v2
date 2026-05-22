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
  const allTimes   = sessions.map(s => s.stats?.totalTimeMs || 0).filter(Boolean);
  const allIous    = sessions.map(s => s.stats?.avgIou).filter(Boolean);
  const allCorrect = sessions.reduce((acc, s) => acc + (s.stats?.correctCount || 0), 0);
  const allImages  = sessions.reduce((acc, s) => acc + (s.stats?.imageCount  || 0), 0);

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
    const { physicianId, mode, completedAt, stats, sessionFeedback } = req.body || {};

    if (!stats || typeof stats !== "object") {
      return res.status(400).json({ ok: false, error: "stats nesnesi gerekli" });
    }

    const sessionId = `${physicianId}_${mode}_${Date.now()}`;
    const record = {
      sessionId,
      mode: mode || "unknown",
      physicianId: physicianId || "anonymous",
      completedAt: completedAt || new Date().toISOString(),
      stats,
      sessionFeedback: sessionFeedback || null,
      comparison: null
    };

    // Check for matching session (opposite mode, same physician)
    const raw = await redis.lrange(SESSIONS_KEY, 0, -1);
    let sessions = raw.map(item => typeof item === "string" ? JSON.parse(item) : item);
    
    const oppositeMode = mode === "solo" ? "ai" : "solo";
    const matchingSession = sessions.find(s => 
      s.physicianId === physicianId && s.mode === oppositeMode
    );

    if (matchingSession && matchingSession.stats) {
      const timeDiff = matchingSession.stats.totalTimeMs - record.stats.totalTimeMs;
      const timeSavingsPct = matchingSession.stats.totalTimeMs > 0 
        ? +((timeDiff / matchingSession.stats.totalTimeMs) * 100).toFixed(2)
        : 0;
      const accuracyGain = +(record.stats.accuracyRate - matchingSession.stats.accuracyRate).toFixed(4);
      const iouGain = record.stats.avgIou && matchingSession.stats.avgIou
        ? +(record.stats.avgIou - matchingSession.stats.avgIou).toFixed(4)
        : null;

      record.comparison = {
        matchedWith: matchingSession.sessionId,
        matchedMode: oppositeMode,
        timeDiffMs: timeDiff,
        timeSavingsPct,
        accuracyGain,
        iouGain,
        timestamp: new Date().toISOString()
      };

      matchingSession.comparison = {
        matchedWith: sessionId,
        matchedMode: mode,
        timeDiffMs: -timeDiff,
        timeSavingsPct: -timeSavingsPct,
        accuracyGain: -accuracyGain,
        iouGain: iouGain ? -iouGain : null,
        timestamp: new Date().toISOString()
      };
    }

    await redis.del(SESSIONS_KEY);
    for (const session of sessions) {
      await redis.rpush(SESSIONS_KEY, JSON.stringify(session));
    }
    await redis.rpush(SESSIONS_KEY, JSON.stringify(record));

    return res.status(201).json({ ok: true, sessionId, stats: record.stats, comparison: record.comparison });
  }

  // ── GET /api/stats ────────────────────────────────────────────────────
  if (req.method === "GET") {
    const raw = await redis.lrange(SESSIONS_KEY, 0, -1);

    let sessions = raw.map(item =>
      typeof item === "string" ? JSON.parse(item) : item
    );

    // Filtreler
    const { mode, physician, summary: summaryOnly } = req.query;
    if (mode)      sessions = sessions.filter(s => s.mode        === mode);
    if (physician) sessions = sessions.filter(s => s.physicianId === physician);

    const output = sessions;

    return res.status(200).json({
      totalSessions: sessions.length,
      aggregate:     buildAggregate(sessions),
      sessions:      output,
    });
  }

  // ── PATCH /api/stats ──────────────────────────────────────────────────
  // Update session feedback
  if (req.method === "PATCH") {
    const { sessionId, sessionFeedback } = req.body || {};

    if (!sessionId) {
      return res.status(400).json({ ok: false, error: "sessionId gerekli" });
    }

    const raw = await redis.lrange(SESSIONS_KEY, 0, -1);
    let sessions = raw.map(item => typeof item === "string" ? JSON.parse(item) : item);

    let updated = false;

    const session = sessions.find(s => s.sessionId === sessionId);
    if (session) {
      session.sessionFeedback = sessionFeedback || null;
      
      await redis.del(SESSIONS_KEY);
      for (const s of sessions) {
        await redis.rpush(SESSIONS_KEY, JSON.stringify(s));
      }
      updated = true;
    }

    return res.status(updated ? 200 : 404).json({
      ok: updated,
      message: updated ? "Feedback kaydedildi" : "Oturum bulunamadı",
    });
  }

  return res.status(405).json({ ok: false, error: "Method not allowed" });
}
