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
 *
 * One extra tool is declared alongside whatever MCP provides:
 * transfer_to_human_agent. It isn't an MCP tool — MCP only exposes Zoho
 * Desk data, not Twilio call control — so it's handled locally instead of
 * being forwarded to mcpBridge.callTool(). Calling it signs a request (see
 * ./hmacAuth.js#signTransfer) and POSTs it back to the Catalyst server that
 * set up this call, asking it to REST-redirect the caller's live call into
 * that number's configured human ring group. Only declared when the
 * opts needed to do that (accountSid/callSid/webhookBaseUrl) are present.
 */

import { GoogleGenAI, Modality } from '@google/genai';
import { connectMcpBridge } from './mcpBridge.js';
import { signTransfer } from './hmacAuth.js';

const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-live-preview';
const DEFAULT_VOICE = process.env.GEMINI_VOICE || 'Puck';
const MCP_SERVER_URL = process.env.MCP_SERVER_URL || '';

const TRANSFER_TOOL_NAME = 'transfer_to_human_agent';
const TRANSFER_TOOL_DECLARATION = {
  name: TRANSFER_TOOL_NAME,
  description: 'Transfers the current call to a human support agent. Use this when the caller explicitly asks for a human, ' +
    'the issue is outside what you can resolve, or per your escalation instructions. Say a brief, friendly hand-off line to ' +
    'the caller BEFORE calling this tool — the call will be redirected shortly after it returns, so anything said after may not be heard.',
  parameters: { type: 'object', properties: {}, required: [] },
};

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
 * @param {string} [opts.accountSid] — needed only for transfer_to_human_agent
 * @param {string} [opts.callSid] — needed only for transfer_to_human_agent
 * @param {string} [opts.to] — the number the caller dialed; tells the
 *   backend which number's ring group to escalate to
 * @param {string} [opts.webhookBaseUrl] — the Catalyst server's own base
 *   URL, so transfer_to_human_agent knows where to call back
 * @returns {Promise<{ sendAudio: (base64Pcm16k: string) => void, close: () => void }>}
 */
export async function openGeminiVoiceSession({ systemInstruction, onAudio, onInterrupted, onTurnComplete, onError, onClose, accountSid, callSid, to, webhookBaseUrl }) {
  const ai = getClient();
  let closed = false;
  const transferAvailable = !!(accountSid && callSid && webhookBaseUrl);

  async function performTransferToHumanAgent() {
    try {
      const { timestamp, signature } = signTransfer(accountSid, callSid);
      const res = await fetch(webhookBaseUrl.replace(/\/$/, '') + '/api/calls/transfer-from-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: JSON.stringify({ accountSid, callSid, to, ts: timestamp, sig: signature }),
      });
      const data = await res.json();
      if (!data.success) return { error: data.message || 'Unable to transfer the call right now.' };
      return { message: 'Transfer initiated — the caller will be connected to a human agent shortly. If you have not already said goodbye, do so now.' };
    } catch (err) {
      console.error('[gemini] transfer_to_human_agent failed:', err.message ?? err);
      return { error: 'Unable to reach the transfer service.' };
    }
  }

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

    const functionResponses = await Promise.all(calls.map(async (call) => {
      if (call.name === TRANSFER_TOOL_NAME) {
        const response = await performTransferToHumanAgent();
        return { id: call.id, name: call.name, response };
      }
      if (!mcpBridge) {
        // An MCP tool was declared but the bridge that serves it is gone
        // (or never connected) — every pending call still needs a response
        // or the Live API blocks the turn (and audio) waiting on it forever.
        return { id: call.id, name: call.name, response: { error: 'Tool service unavailable.' } };
      }
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
      ...(() => {
        const functionDeclarations = [
          ...(mcpBridge ? mcpBridge.functionDeclarations : []),
          ...(transferAvailable ? [TRANSFER_TOOL_DECLARATION] : []),
        ];
        return functionDeclarations.length ? { tools: [{ functionDeclarations }] } : {};
      })(),
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
