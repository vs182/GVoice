/**
 * geminiVoiceAgent.js — Thin wrapper around @google/genai's Live API for a
 * single phone call's AI-agent session.
 *
 * One GEMINI_API_KEY for the whole platform (not per-tenant) — matches how
 * the other third-party secrets in this app (Zoho) are configured, and is
 * the right size for "one admin's API key powering the feature" rather than
 * building out per-tenant key storage for a v1. If this becomes something
 * customers bring their own key for, it needs the same AES-at-rest treatment
 * Twilio credentials already get (see services/crypto.js) — don't store a
 * customer-supplied Gemini key in plaintext.
 *
 * Wire format (verified against Google's docs before writing this, not
 * guessed):
 *   input  → session.sendRealtimeInput({ audio: { data: base64PCM16,
 *            mimeType: 'audio/pcm;rate=16000' } })   — 16-bit PCM, 16kHz
 *   output ← message.serverContent.modelTurn.parts[].inlineData.data
 *            — 16-bit PCM, 24kHz, base64 (fixed, not configurable)
 *   barge-in ← message.serverContent.interrupted === true
 *
 * Automatic voice-activity detection is Gemini's default when you just
 * stream continuous audio without manual turn markers — exactly the "caller
 * talks, pauses, agent responds" shape a phone call needs, so there's no
 * custom VAD here.
 */

import { GoogleGenAI, Modality } from '@google/genai';

const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-live-preview';
const DEFAULT_VOICE = process.env.GEMINI_VOICE || 'Puck';

let _ai = null;
function getClient() {
  if (_ai) return _ai;
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set');
  _ai = new GoogleGenAI({ apiKey });
  return _ai;
}

/**
 * Opens one Live API session for one phone call.
 *
 * @param {object} opts
 * @param {string} opts.systemInstruction — the agent's persona/instructions
 *   (an IVR menu item's `value`, or a sane default).
 * @param {(base64Pcm24k: string) => void} opts.onAudio — called per output
 *   audio chunk, still in Gemini's native 24kHz PCM16 — callers resample.
 * @param {() => void} opts.onInterrupted — caller barged in; stop/flush
 *   whatever's currently queued for playback.
 * @param {(err: Error) => void} opts.onError
 * @param {() => void} [opts.onClose]
 * @returns {Promise<{ sendAudio: (base64Pcm16k: string) => void, close: () => void }>}
 */
export async function openGeminiVoiceSession({ systemInstruction, onAudio, onInterrupted, onError, onClose }) {
  const ai = getClient();
  let closed = false;

  const session = await ai.live.connect({
    model: DEFAULT_MODEL,
    config: {
      responseModalities: [Modality.AUDIO],
      systemInstruction: { parts: [{ text: systemInstruction }] },
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: DEFAULT_VOICE } },
      },
    },
    callbacks: {
      onopen: () => console.log('[gemini] session opened'),
      onmessage: (message) => {
        // TEMPORARY instrumentation to locate a "session opens, no audio
        // ever comes back" report — logs every message shape so we can see
        // whether Gemini is engaging with the call at all (setupComplete,
        // turnComplete with no audio, etc.) before it's removed again.
        const content = message?.serverContent;
        if (!content) {
          console.log('[gemini] non-serverContent message:', JSON.stringify(message).slice(0, 300));
          return;
        }

        if (content.interrupted && typeof onInterrupted === 'function') {
          onInterrupted();
        }

        const parts = content.modelTurn?.parts ?? [];
        const audioParts = parts.filter((p) => p.inlineData?.data);
        if (audioParts.length === 0) {
          console.log('[gemini] serverContent with no audio — turnComplete:', content.turnComplete,
            'generationComplete:', content.generationComplete, 'interrupted:', content.interrupted,
            'partsCount:', parts.length, 'partTypes:', parts.map((p) => Object.keys(p)));
        }
        for (const part of audioParts) {
          onAudio?.(part.inlineData.data);
        }
      },
      onerror: (err) => {
        console.error('[gemini] session error:', err?.message ?? err);
        if (typeof onError === 'function') onError(err instanceof Error ? err : new Error(String(err)));
      },
      onclose: () => {
        console.log('[gemini] session closed');
        if (!closed && typeof onClose === 'function') onClose();
        closed = true;
      },
    },
  });

  return {
    sendAudio(base64Pcm16k) {
      if (closed) return;
      try {
        session.sendRealtimeInput({ audio: { data: base64Pcm16k, mimeType: 'audio/pcm;rate=16000' } });
      } catch (err) {
        console.warn('[gemini] sendAudio failed:', err.message ?? err);
      }
    },
    close() {
      if (closed) return;
      closed = true;
      try { session.close(); } catch (_) {}
    },
  };
}
