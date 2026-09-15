/**
 * mediaBridge.js — Bridges a Twilio Media Streams WebSocket connection to a
 * Gemini Live API session, translating audio formats in both directions in
 * real time. This is the standalone (Render-hosted) counterpart of what
 * used to live at twilio-catalyst-server/services/mediaStreamServer.js —
 * moved here because Zoho Catalyst's gateway doesn't pass WebSocket
 * upgrades through to AppSail (confirmed live), so the persistent
 * call-length connection Twilio's Media Streams needs has to live
 * somewhere else. Every other webhook (`/webhook/voice`, `/webhook/incoming`,
 * etc.) stays on Catalyst — those are plain HTTP and work fine there.
 *
 * Twilio → us:  {event:"connected"|"start"|"media"|"stop", ...}
 * us → Twilio:  {event:"media", streamSid, media:{payload}} and
 *               {event:"clear", streamSid} (barge-in)
 *
 * Tenant/prompt data arrives via the `start` event's `customParameters` —
 * confirmed live that a query string on the Stream url is NOT reliably
 * delivered (arrived as null on a real call); <Parameter> child elements
 * are Twilio's actual mechanism. That also means nothing is known about the
 * call until `start` arrives — inbound `media` frames that arrive before
 * the Gemini session finishes opening are buffered, not dropped.
 *
 * See services/hmacAuth.js for how a connection's accountSid/prompt are
 * verified as having actually come from the Catalyst server, since this
 * service has no independent way to check "is this a real tenant" — it
 * deliberately carries no Catalyst DataStore access.
 */

import { WebSocketServer } from 'ws';
import { verify } from './hmacAuth.js';
import { openGeminiVoiceSession } from './geminiVoiceAgent.js';
import { twilioPayloadToGeminiPcm16, geminiPcm16ToTwilioPayload, chunkMulawInto20msFrames } from './audioCodec.js';

const DEFAULT_SYSTEM_INSTRUCTION =
  'You are a helpful, concise phone support agent. Keep responses short and ' +
  'conversational — this is a live phone call, not a chat window. Greet the ' +
  'caller warmly and ask how you can help.';

// Bound cost exposure from a stuck/abandoned call — nothing in Twilio's
// Media Streams protocol guarantees a `stop` event always arrives promptly.
const MAX_SESSION_MS = 10 * 60 * 1000;

// Safety net for automatic VAD going quiet mid-call (observed live on this
// preview model: real caller audio kept flowing in for 24+ seconds with
// zero response — no audio, no turnComplete, nothing — until the caller
// gave up). If Gemini hasn't produced anything in this long, proactively
// nudge it to check in rather than let the call sit in dead silence.
const SILENCE_NUDGE_MS  = 12_000;
const WATCHDOG_TICK_MS  = 4_000;

export function attachMediaStreamServer(httpServer) {
  const wss = new WebSocketServer({ server: httpServer, path: '/media-stream' });

  wss.on('connection', (twilioWs) => {
    handleConnection(twilioWs).catch((err) => {
      console.error('[media-stream] connection handler failed:', err.message ?? err);
      try { twilioWs.close(); } catch (_) {}
    });
  });

  console.log('[media-stream] WebSocket server attached at /media-stream');
  return wss;
}

async function handleConnection(twilioWs) {
  console.log('[media-stream] connection opened');

  let streamSid    = null;
  let geminiSession = null;
  let geminiReady   = false;
  let starting      = false; // true while verifying + opening Gemini session
  let greetingSettled = false; // true once the greeting's first turn completes
  let pendingIn     = [];    // caller audio queued until it's safe to forward
  let sessionTimer  = null;
  let watchdogTimer = null;
  let lastActivityAt = null; // last time Gemini actually produced audio
  let closed        = false;

  // Counters for the one-line summary logged when the call ends.
  let inboundFrameCount  = 0;
  let geminiAudioChunks  = 0;
  let outboundFrameCount = 0;

  function sendToTwilio(base64MulawPayload) {
    outboundFrameCount++;
    if (!streamSid || twilioWs.readyState !== twilioWs.OPEN) {
      console.warn('[media-stream] DROPPED outbound frame — streamSid:', streamSid, 'wsState:', twilioWs.readyState);
      return;
    }
    twilioWs.send(JSON.stringify({
      event: 'media',
      streamSid,
      media: { payload: base64MulawPayload },
    }));
  }

  function clearTwilioPlayback() {
    if (!streamSid || twilioWs.readyState !== twilioWs.OPEN) return;
    twilioWs.send(JSON.stringify({ event: 'clear', streamSid }));
  }

  function cleanup() {
    if (closed) return;
    closed = true;
    if (sessionTimer) clearTimeout(sessionTimer);
    if (watchdogTimer) clearInterval(watchdogTimer);
    if (geminiSession) geminiSession.close();
    try { twilioWs.close(); } catch (_) {}
    console.log('[media-stream] cleaned up, streamSid:', streamSid);
  }

  async function handleStart(msg) {
    starting  = true;
    streamSid = msg.start?.streamSid ?? msg.streamSid;
    const p = msg.start?.customParameters ?? {};

    console.log('[media-stream] call started, streamSid:', streamSid, 'callSid:', msg.start?.callSid, 'accountSid:', p.accountSid);

    if (!verify(p.accountSid, p.ts, p.sig)) {
      console.warn('[media-stream] rejecting — signature invalid or stale for accountSid:', p.accountSid);
      cleanup();
      return;
    }

    geminiSession = await openGeminiVoiceSession({
      systemInstruction: p.prompt || DEFAULT_SYSTEM_INSTRUCTION,
      onAudio: (base64Pcm24k) => {
        geminiAudioChunks++;
        lastActivityAt = Date.now();
        const mulawPayload = geminiPcm16ToTwilioPayload(base64Pcm24k);
        const mulawBuffer  = Buffer.from(mulawPayload, 'base64');
        chunkMulawInto20msFrames(mulawBuffer).forEach(sendToTwilio);
      },
      onInterrupted: () => {
        // Caller barged in — stop whatever Twilio still has queued for playback.
        clearTwilioPlayback();
      },
      onTurnComplete: () => {
        if (greetingSettled) return; // only care about the very first one
        greetingSettled = true;
        console.log('[media-stream] greeting turn complete — now forwarding live caller audio, buffered frames:', pendingIn.length);
        if (pendingIn.length) {
          pendingIn.forEach((payload) => geminiSession.sendAudio(twilioPayloadToGeminiPcm16(payload)));
          pendingIn = [];
        }
      },
      onError: (err) => {
        console.error('[media-stream] Gemini session error:', err.message);
        cleanup();
      },
      onClose: cleanup,
    });

    if (closed) { geminiSession.close(); return; } // cleanup() ran while we were awaiting connect()

    geminiReady    = true;
    starting       = false;
    lastActivityAt = Date.now(); // starts the silence clock from session-open, giving the greeting a fair window

    sessionTimer = setTimeout(() => {
      console.warn('[media-stream] max session duration reached, closing:', streamSid);
      cleanup();
    }, MAX_SESSION_MS);

    watchdogTimer = setInterval(() => {
      if (closed || !lastActivityAt) return;
      const silentFor = Date.now() - lastActivityAt;
      if (silentFor >= SILENCE_NUDGE_MS) {
        console.warn('[media-stream] no Gemini activity for', silentFor, 'ms — sending recovery nudge, streamSid:', streamSid);
        geminiSession.sendNudge('There has been a long pause on the call. Politely check in — ask if the caller is still there or if they need anything else.');
        lastActivityAt = Date.now(); // don't nudge again every tick while still waiting on this one
      }
    }, WATCHDOG_TICK_MS);
  }

  twilioWs.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    switch (msg.event) {
      case 'start':
        handleStart(msg).catch((err) => {
          console.error('[media-stream] start handling failed:', err.message ?? err);
          cleanup();
        });
        break;

      case 'media':
        if (!msg.media?.payload) break;
        inboundFrameCount++;
        if (geminiReady && greetingSettled) {
          geminiSession.sendAudio(twilioPayloadToGeminiPcm16(msg.media.payload));
        } else if (starting || (geminiReady && !greetingSettled)) {
          // Buffered, not forwarded, until the greeting's first turn
          // completes — forwarding live caller audio into an in-progress
          // turn that hasn't produced any audio yet is exactly what let
          // automatic VAD interrupt the greeting before it ever spoke,
          // confirmed live: a real call's greeting was interrupted at the
          // ~6s mark with zero audio ever generated for it.
          pendingIn.push(msg.media.payload);
        }
        break;

      case 'stop':
        console.log('[media-stream] call stopped, streamSid:', streamSid,
          '| totals — inbound frames:', inboundFrameCount, 'gemini audio chunks:', geminiAudioChunks, 'outbound frames:', outboundFrameCount);
        cleanup();
        break;

      default:
        break; // 'connected' and any future event types — nothing to do
    }
  });

  twilioWs.on('close', cleanup);
  twilioWs.on('error', (err) => {
    console.warn('[media-stream] Twilio WS error:', err.message);
    cleanup();
  });
}
