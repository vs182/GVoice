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
 *
 * Tool calling: if MCP_SERVER_URL is set, this connects to that MCP server
 * (see ./mcpBridge.js) and declares its tools to the Live session as
 * `functionDeclarations` — Live API has no native MCP support, only plain
 * function declarations, so the bridge module does that translation by
 * hand. `message.toolCall` arrives as a distinct message type alongside
 * `message.serverContent`, not inside it; the Live API blocks the turn
 * (including audio) until every functionCall in it gets a matching
 * functionResponse by id, so handleToolCall() always responds — with an
 * error payload if the tool itself failed — never silently drops one.
 */

import { GoogleGenAI, Modality } from '@google/genai';
import { connectMcpBridge } from './mcpBridge.js';

const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-live-preview';
const DEFAULT_VOICE = process.env.GEMINI_VOICE || 'Puck';
const MCP_SERVER_URL = process.env.MCP_SERVER_URL || '';

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
 * @param {() => void} [opts.onTurnComplete] — fires whenever a model turn
 *   finishes (the greeting, a nudge, a normal reply — every one). Callers
 *   typically only care about the FIRST one, to know when it's safe to
 *   start forwarding live caller audio without risking an interrupt firing
 *   on a turn that hasn't produced any audio yet — see mediaBridge.js.
 * @param {(err: Error) => void} opts.onError
 * @param {() => void} [opts.onClose]
 * @returns {Promise<{ sendAudio: (base64Pcm16k: string) => void, close: () => void }>}
 */
export async function openGeminiVoiceSession({ systemInstruction, onAudio, onInterrupted, onTurnComplete, onError, onClose }) {
  const ai = getClient();
  let closed = false;

  // A mutable reference `onmessage`'s toolCall handling closes over, rather
  // than the `session` const itself — `onmessage` is wired up as part of the
  // very call that produces `session`, so referencing that const directly
  // from inside it would hit the same temporal-dead-zone trap the greeting
  // trigger below already had to work around. This is assigned once the
  // `await` resolves, further down.
  let liveSession = null;

  // Best-effort: an MCP server unreachable/misconfigured shouldn't block a
  // phone call from connecting — it just proceeds without tools.
  let mcpBridge = null;
  if (MCP_SERVER_URL) {
    try {
      mcpBridge = await connectMcpBridge(MCP_SERVER_URL);
    } catch (err) {
      console.error('[mcp] failed to connect, continuing without tools:', err.message ?? err);
    }
  }

  async function handleToolCall(toolCall) {
    const calls = toolCall?.functionCalls ?? [];
    if (!calls.length || !liveSession) return;

    if (!mcpBridge) {
      // Tools were declared but the bridge that serves them is gone (or
      // never connected) — every pending call still needs a response or
      // the Live API blocks the turn (and audio) waiting on it forever.
      liveSession.sendToolResponse({
        functionResponses: calls.map((c) => ({ id: c.id, name: c.name, response: { error: 'Tool service unavailable.' } })),
      });
      return;
    }

    const functionResponses = await Promise.all(calls.map(async (call) => {
      try {
        const result = await mcpBridge.callTool(call.name, call.args);
        return { id: call.id, name: call.name, response: { result } };
      } catch (err) {
        console.error('[mcp] tool call failed:', call.name, err.message ?? err);
        return { id: call.id, name: call.name, response: { error: err.message ?? String(err) } };
      }
    }));

    try {
      liveSession.sendToolResponse({ functionResponses });
    } catch (err) {
      console.warn('[gemini] sendToolResponse failed:', err.message ?? err);
    }
  }

  const session = await ai.live.connect({
    model: DEFAULT_MODEL,
    config: {
      responseModalities: [Modality.AUDIO],
      systemInstruction: { parts: [{ text: systemInstruction }] },
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: DEFAULT_VOICE } },
      },
      ...(mcpBridge && mcpBridge.functionDeclarations.length
        ? { tools: [{ functionDeclarations: mcpBridge.functionDeclarations }] }
        : {}),
    },
    callbacks: {
      onopen: () => console.log('[gemini] session opened'),
      onmessage: (message) => {
        if (message?.toolCall) {
          handleToolCall(message.toolCall).catch((err) => console.error('[mcp] tool call handling failed:', err.message ?? err));
          return;
        }

        const content = message?.serverContent;
        if (!content) return; // setupComplete, sessionResumptionUpdate, etc. — nothing to do

        if (content.interrupted) {
          console.log('[gemini] turn interrupted (caller barge-in)');
          onInterrupted?.();
        }

        const audioParts = (content.modelTurn?.parts ?? []).filter((p) => p.inlineData?.data);
        for (const part of audioParts) {
          onAudio?.(part.inlineData.data);
        }
        if (content.turnComplete) onTurnComplete?.();
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
  liveSession = session;

  // Automatic VAD only ever REACTS to detected speech — it never speaks
  // first. On a real call, the caller is waiting to hear the agent greet
  // them, the same as any normal call; if nothing ever prompts a turn, the
  // whole call can go by in silence even though audio is flowing into
  // Gemini the entire time (confirmed live: a real 28s call sent 1425 real
  // audio frames and never got a single serverContent message back).
  // Kicking off one explicit turn here — safely after `session` exists,
  // not inside onopen, which fires before this `await` resolves and would
  // hit a temporal-dead-zone error referencing `session` early — covers
  // that. sendRealtimeInput above still handles the rest of the
  // conversation reactively once this first exchange completes.
  try {
    session.sendClientContent({ turns: 'The caller has just connected. Greet them warmly in one short sentence and ask how you can help.' });
  } catch (err) {
    console.warn('[gemini] initial greeting trigger failed:', err.message ?? err);
  }

  return {
    sendAudio(base64Pcm16k) {
      if (closed) return;
      try {
        session.sendRealtimeInput({ audio: { data: base64Pcm16k, mimeType: 'audio/pcm;rate=16000' } });
      } catch (err) {
        console.warn('[gemini] sendAudio failed:', err.message ?? err);
      }
    },
    // Explicit text turn — same mechanism as the initial greeting above.
    // Used as a recovery nudge when automatic VAD goes quiet for too long
    // mid-call (observed live on this preview model: real audio kept
    // flowing in but the server produced nothing at all — no audio, no
    // turnComplete, nothing — for 24+ seconds until the caller gave up and
    // hung up). Not a fix for whatever causes VAD to stop reacting, but a
    // safety net so a stuck session doesn't just sit in dead silence.
    sendNudge(text) {
      if (closed) return;
      try {
        session.sendClientContent({ turns: text });
      } catch (err) {
        console.warn('[gemini] sendNudge failed:', err.message ?? err);
      }
    },
    close() {
      if (closed) return;
      closed = true;
      try { session.close(); } catch (_) {}
      if (mcpBridge) mcpBridge.close().catch(() => {});
    },
  };
}
