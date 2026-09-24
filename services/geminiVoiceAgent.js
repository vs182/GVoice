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
 * Two extra tools are declared alongside whatever MCP provides — neither is
 * an MCP tool (MCP only exposes Zoho Desk data, not call control), so both
 * are handled locally instead of being forwarded to mcpBridge.callTool():
 *
 *   transfer_to_human_agent — signs a request (see ./hmacAuth.js#signTransfer)
 *     and POSTs it to the Catalyst server, asking whether any agent
 *     configured for this number is actually *available* right now (real
 *     presence, not just configured — see ztwilio's services/agentPresence.js).
 *     If so, the Catalyst server records what to ring once this call ends
 *     and this schedules the session to close after the model's next turn
 *     completes (giving it room to say goodbye first) — Twilio's <Connect>
 *     action fires on that close and does the actual ringing (a real,
 *     documented mechanism; earlier versions of this REST-redirected the
 *     live call directly, which isn't documented-safe for a call mid-Dial
 *     or mid-Connect and reliably produced "an application error has
 *     occurred"). If no one's available, nothing is scheduled — the
 *     functionResponse tells the model to handle it conversationally
 *     instead (e.g. offer a callback via its own ZohoDesk_createTask tool).
 *
 *   end_call — the model's own way to hang up once a conversation is
 *     genuinely finished, rather than leaving the caller on an open line
 *     indefinitely. Also just schedules a close-after-next-turn — never
 *     closes mid-sentence.
 */

import { GoogleGenAI, Modality } from '@google/genai';
import { connectMcpBridge } from './mcpBridge.js';
import { signTransfer } from './hmacAuth.js';

const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-live-preview';
const DEFAULT_VOICE = process.env.GEMINI_VOICE || 'Puck';
const MCP_SERVER_URL = process.env.MCP_SERVER_URL || '';

// Give the last turn's audio time to actually reach Twilio (and get played)
// before the WebSocket that's carrying it closes — sendAudio() calls having
// returned only means the bytes were handed off, not that they've arrived.
const CLOSE_AFTER_TURN_DELAY_MS = 1200;

const TRANSFER_TOOL_NAME = 'transfer_to_human_agent';
const TRANSFER_TOOL_DECLARATION = {
  name: TRANSFER_TOOL_NAME,
  description: 'Checks whether a human agent is available and, if so, arranges to transfer the call to them once you finish ' +
    'speaking. Use this when the caller explicitly asks for a human, the issue is outside what you can resolve, or per your ' +
    'escalation instructions. The tool result tells you whether anyone was actually available — if yes, say a brief, friendly ' +
    'hand-off line and nothing further; if no, apologize and offer another way to help (e.g. logging a callback request) instead.',
  parameters: { type: 'object', properties: {}, required: [] },
};

const END_CALL_TOOL_NAME = 'end_call';
const END_CALL_TOOL_DECLARATION = {
  name: END_CALL_TOOL_NAME,
  description: 'Ends the current call. You MUST call this in the exact same turn as any goodbye ("goodbye", "have a good day", ' +
    '"take care", etc.) — saying goodbye out loud without calling this function does NOT end the call, it just leaves the ' +
    'caller sitting on an open line waiting. Never say a farewell word unless you are also calling this tool right after it ' +
    'in that same turn. Use this once the conversation is genuinely finished — the issue is resolved or the caller says they ' +
    'are done.',
  parameters: { type: 'object', properties: {}, required: [] },
};

// The model is instructed above to always pair a farewell with an end_call
// tool call in the same turn, but observed live: it sometimes says "goodbye"
// and just... keeps the call open anyway, leaving the caller hanging (worse,
// the silence-recovery nudge in mediaBridge.js then prompts it to speak
// again, undoing the goodbye entirely). This regex-based fallback treats a
// spoken farewell as equivalent to an actual end_call invocation, so the
// call still closes even when the model forgets the tool call itself.
const FAREWELL_RE = /\bgoodbye\b|\bgood bye\b/i;

let _ai = null;
function getClient() {
  if (_ai) return _ai;
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set');
  _ai = new GoogleGenAI({ apiKey });
  return _ai;
}

/**
 * Post-call Desk logging — every AI-handled call gets a Zoho Desk Event (with
 * the full transcript as a comment) and, if the transcript looks like it
 * needs human follow-up, a Task too. This runs from close() below, which
 * fires no matter how the call ended (caller hangup, the model's own
 * end_call/transfer tools, an error, the max-session timeout) — deliberately
 * NOT left to the model to remember to do via its own tool calls, since a
 * caller who just hangs up gives the model no final turn to act in.
 */

// Zoho Desk API's create-record endpoints use inconsistent response
// wrappings across MCP tools (and the MCP SDK itself sometimes carries the
// payload as a JSON string in `content` rather than `structuredContent`) —
// try every shape seen in practice rather than assuming one.
function extractId(mcpResult) {
  const direct = mcpResult?.structuredContent?.data?.id
    || mcpResult?.structuredContent?.id
    || mcpResult?.structuredContent?.data?.data?.id;
  if (direct) return direct;
  const text = mcpResult?.content?.[0]?.text;
  if (text) {
    try {
      const parsed = JSON.parse(text);
      return parsed?.id || parsed?.data?.id || null;
    } catch (_) { /* not JSON */ }
  }
  return null;
}

function extractList(mcpResult) {
  const list = mcpResult?.structuredContent?.data?.data || mcpResult?.structuredContent?.data;
  if (Array.isArray(list)) return list;
  const text = mcpResult?.content?.[0]?.text;
  if (text) {
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed?.data)) return parsed.data;
      if (Array.isArray(parsed)) return parsed;
    } catch (_) { /* not JSON */ }
  }
  return [];
}

// Zoho Desk has no single "the" default department — cached per process
// (this service is long-lived on Render, and the org's departments change
// rarely) rather than refetched every call.
let cachedDepartmentId = null;
async function resolveDepartmentId(mcpBridge) {
  if (cachedDepartmentId) return cachedDepartmentId;
  try {
    // No isEnabled filter — this org's departments came back empty under
    // that filter (confirmed live: "no department id available"), so just
    // take whichever department exists first. Any real department is fine
    // here; this is only ever used to satisfy Desk's required field on
    // Event/Task creation, not to route the call anywhere.
    const result = await mcpBridge.callTool('ZohoDesk_getDepartments', { query_params: { limit: 1 } });
    const first = extractList(result)[0];
    if (first?.id) cachedDepartmentId = first.id;
    else console.error('[call-log] getDepartments returned no departments:', JSON.stringify(result));
  } catch (err) {
    console.error('[call-log] getDepartments failed:', err.message ?? err);
  }
  return cachedDepartmentId;
}

// Deterministic follow-up check — deliberately NOT a second Gemini call.
// That was tried first and turned out unreliable in practice (503s under
// load silently killed every Task); this has no external dependency, so it
// can't fail. Matched only against the CALLER's own words, since escalation/
// frustration signal comes from them, not the agent paraphrasing it back.
const FRUSTRATION_RE = /\b(angry|frustrat(?:ed|ing)|annoyed|upset|unacceptable|ridiculous|terrible|awful|worst|fed up|sick of|not happy|unhappy|disappointed|furious)\b/i;
const PRIORITY_RE = /\b(urgent|immediately|asap|escalate|escalation|priority|as soon as possible|right away|emergency|critical|manager|supervisor|within \d+\s*(?:hour|hours|minute|minutes|day|days)|call\s*(?:me\s*)?back|callback|follow[\s-]?up|complaint)\b/i;

function checkNeedsFollowUp(transcriptLog) {
  const callerText = transcriptLog.filter((t) => t.role === 'caller').map((t) => t.text).join(' ');
  const labels = [];
  if (FRUSTRATION_RE.test(callerText)) labels.push('Frustrated Caller');
  if (PRIORITY_RE.test(callerText)) labels.push('Priority Request');
  return { needsFollowUp: labels.length > 0, queryName: labels.join(' / ') };
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
 * @param {() => void} [opts.onActivity] — fires on non-audio signs of life
 *   (currently: a tool call starting) so a caller's own silence/stuck-session
 *   watchdog doesn't mistake normal tool-call latency for a dead session.
 * @param {(err: Error) => void} opts.onError
 * @param {() => void} [opts.onClose]
 * @param {() => void} [opts.onCloseRequested] — the transfer/end-call tools'
 *   way of asking to hang up, fired once shortly after the turn following
 *   that tool call finishes (see CLOSE_AFTER_TURN_DELAY_MS) — callers
 *   should treat this exactly like onClose (run cleanup, close the Twilio
 *   WebSocket), just from a different trigger.
 * @param {string} [opts.accountSid] — needed only for transfer_to_human_agent
 * @param {string} [opts.callSid] — needed only for transfer_to_human_agent
 * @param {string} [opts.to] — the number the caller dialed; tells the
 *   backend which number's ring group to escalate to
 * @param {string} [opts.webhookBaseUrl] — the Catalyst server's own base
 *   URL, so transfer_to_human_agent knows where to call back
 * @returns {Promise<{ sendAudio: (base64Pcm16k: string) => void, close: () => void }>}
 */
export async function openGeminiVoiceSession({ systemInstruction, onAudio, onInterrupted, onTurnComplete, onActivity, onError, onClose, onCloseRequested, accountSid, callSid, to, webhookBaseUrl, agentName, callerNumber, contactId, contactName }) {
  const ai = getClient();
  let closed = false;
  const transferAvailable = !!(accountSid && callSid && webhookBaseUrl);
  const callStartedAt = Date.now();

  // Running transcript, built from Gemini's own transcription of both sides
  // — independent of anything the model does — so post-call Desk logging
  // never depends on the model remembering to summarize itself. Consecutive
  // chunks from the same speaker are merged; a chunk from the other speaker
  // starts a new entry.
  const transcriptLog = [];
  function appendTranscript(role, text) {
    if (!text) return;
    const last = transcriptLog[transcriptLog.length - 1];
    if (last && last.role === role) last.text += text;
    else transcriptLog.push({ role, text });
  }

  // Accumulates just the CURRENT turn's agent speech, reset every
  // turnComplete — used only for the farewell-detection fallback below, kept
  // separate from transcriptLog (which merges across turn boundaries).
  let currentTurnAgentText = '';

  async function finalizeCallLogging() {
    if (!mcpBridge) {
      console.warn('[call-log] no MCP bridge for this session, skipping Event/Task log');
      return;
    }
    if (!transcriptLog.length) {
      console.warn('[call-log] transcript is empty (nothing was transcribed), skipping Event/Task log');
      return;
    }
    const transcriptText = transcriptLog
      .map((t) => (t.role === 'caller' ? 'Caller: ' : 'Agent: ') + t.text.trim())
      .join('\n');

    let resolvedContactId = contactId;
    if (!resolvedContactId && callerNumber) {
      try {
        const created = await mcpBridge.callTool('ZohoDesk_createContact', { body: { lastName: callerNumber, phone: callerNumber } });
        resolvedContactId = extractId(created);
        if (!resolvedContactId) console.error('[call-log] createContact returned no extractable id:', JSON.stringify(created));
      } catch (err) {
        console.error('[call-log] createContact failed:', err.message ?? err);
      }
    }
    if (!resolvedContactId) {
      console.warn('[call-log] no contact id available, skipping Event/Task log');
      return;
    }

    const departmentId = await resolveDepartmentId(mcpBridge);
    if (!departmentId) {
      console.warn('[call-log] no department id available, skipping Event/Task log');
      return;
    }

    const displayName = contactName || callerNumber || 'Unknown caller';
    const durationSeconds = Math.max(1, Math.round((Date.now() - callStartedAt) / 1000));

    let eventId = null;
    try {
      const eventResult = await mcpBridge.callTool('ZohoDesk_createEvent', {
        body: {
          subject: `Answered by ${agentName || 'AI Agent'} for ${displayName}`.slice(0, 300),
          contactId: resolvedContactId,
          departmentId,
          startTime: new Date(callStartedAt).toISOString(),
          duration: String(durationSeconds),
          description: 'AI agent phone call — full transcript in comments.',
          status: 'Completed',
        },
      });
      eventId = extractId(eventResult);
      if (!eventId) console.error('[call-log] createEvent returned no extractable id:', JSON.stringify(eventResult));
      else console.log('[call-log] created Event', eventId, 'for', displayName);
    } catch (err) {
      console.error('[call-log] createEvent failed:', err.message ?? err);
    }
    if (!eventId) return;

    try {
      await mcpBridge.callTool('ZohoDesk_createEventComment', {
        path_variables: { eventId },
        body: { content: transcriptText.slice(0, 32000), contentType: 'plainText' },
      });
      console.log('[call-log] added transcript comment to Event', eventId);
    } catch (err) {
      console.error('[call-log] createEventComment failed:', err.message ?? err);
    }

    // Task only for calls that actually look like they need follow-up —
    // priority/escalation language or caller frustration, checked
    // deterministically (see checkNeedsFollowUp above), not via a second
    // Gemini call (that was tried first and was unreliable under load).
    const followUp = checkNeedsFollowUp(transcriptLog);
    if (!followUp.needsFollowUp) {
      console.log('[call-log] no priority/frustration signal detected, skipping Task');
      return;
    }
    try {
      const taskResult = await mcpBridge.callTool('ZohoDesk_createTask', {
        body: {
          subject: `${followUp.queryName} - ${agentName || 'AI Agent'} - ${displayName}`.slice(0, 300),
          description: transcriptText.slice(0, 65535),
          contactId: resolvedContactId,
          departmentId,
          priority: 'High',
          status: 'Not Started',
        },
      });
      const taskId = extractId(taskResult);
      if (!taskId) console.error('[call-log] createTask returned no extractable id:', JSON.stringify(taskResult));
      else console.log('[call-log] created Task', taskId, 'for', displayName, '-', followUp.queryName);
    } catch (err) {
      console.error('[call-log] createTask failed:', err.message ?? err);
    }
  }

  // Set by a transfer/end-call tool once it decides the session should wind
  // down — checked after the NEXT turn completes (giving the model room to
  // actually say its goodbye first) rather than closing the instant the
  // tool call itself resolves, which could cut that goodbye off mid-word.
  let closeAfterNextTurn = false;

  async function performTransferToHumanAgent() {
    try {
      const { timestamp, signature } = signTransfer(accountSid, callSid);
      const res = await fetch(webhookBaseUrl.replace(/\/$/, '') + '/api/calls/transfer-from-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: JSON.stringify({ accountSid, callSid, to, ts: timestamp, sig: signature }),
      });
      const data = await res.json();
      if (!data.success) return { error: data.message || 'Unable to check agent availability right now.' };
      if (!data.agentsAvailable) {
        return { agentsAvailable: false, message: 'No human agent is available right now. Do not end the call — apologize and offer another way to help, such as logging a callback request.' };
      }
      closeAfterNextTurn = true;
      return { agentsAvailable: true, message: 'An agent is available. Say a brief, friendly goodbye now — the call will transfer right after this turn, so say nothing further after your goodbye.' };
    } catch (err) {
      console.error('[gemini] transfer_to_human_agent failed:', err.message ?? err);
      return { error: 'Unable to reach the transfer service.' };
    }
  }

  function performEndCall() {
    console.log('[gemini] end_call tool invoked by model');
    closeAfterNextTurn = true;
    return { message: 'Ending the call after this turn.' };
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
      if (call.name === END_CALL_TOOL_NAME) {
        return { id: call.id, name: call.name, response: performEndCall() };
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
      // Powers the post-call transcript (see finalizeCallLogging above) —
      // independent of audio playback, so it works even if the call ends
      // before the model ever calls a tool of its own.
      inputAudioTranscription: {},
      outputAudioTranscription: {},
      ...(() => {
        const functionDeclarations = [
          ...(mcpBridge ? mcpBridge.functionDeclarations : []),
          ...(transferAvailable ? [TRANSFER_TOOL_DECLARATION] : []),
          END_CALL_TOOL_DECLARATION,
        ];
        return functionDeclarations.length ? { tools: [{ functionDeclarations }] } : {};
      })(),
    },
    callbacks: {
      onopen: () => console.log('[gemini] session opened'),
      onmessage: (message) => {
        if (message?.toolCall) {
          // A tool call (e.g. a Zoho Desk lookup) can legitimately take a
          // few seconds with no audio output at all — without this, the
          // silence-recovery watchdog in mediaBridge.js (which only sees
          // audio as "activity") mistakes that wait for a stuck session and
          // interrupts with a premature "are you still there?" right before
          // the real answer would have arrived.
          onActivity?.();
          handleToolCall(message.toolCall).catch((err) => console.error('[mcp] tool call handling failed:', err.message ?? err));
          return;
        }

        const content = message?.serverContent;
        if (!content) return; // setupComplete, sessionResumptionUpdate, etc. — nothing to do

        if (content.inputTranscription?.text) appendTranscript('caller', content.inputTranscription.text);
        if (content.outputTranscription?.text) {
          appendTranscript('agent', content.outputTranscription.text);
          currentTurnAgentText += content.outputTranscription.text;
        }

        if (content.interrupted) {
          console.log('[gemini] turn interrupted (caller barge-in)');
          onInterrupted?.();
        }

        const audioParts = (content.modelTurn?.parts ?? []).filter((p) => p.inlineData?.data);
        for (const part of audioParts) {
          onAudio?.(part.inlineData.data);
        }
        if (content.turnComplete) {
          onTurnComplete?.();
          if (!closeAfterNextTurn && FAREWELL_RE.test(currentTurnAgentText)) {
            console.log('[gemini] farewell detected without an end_call tool invocation — closing anyway');
            closeAfterNextTurn = true;
          }
          currentTurnAgentText = '';
          if (closeAfterNextTurn) {
            closeAfterNextTurn = false;
            setTimeout(() => {
              if (!closed) onCloseRequested?.();
            }, CLOSE_AFTER_TURN_DELAY_MS);
          }
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
      // Fire-and-forget: audio is already done, so nothing is waiting on
      // this, but it must finish using mcpBridge before closing it.
      finalizeCallLogging()
        .catch((err) => console.error('[call-log] finalize failed:', err.message ?? err))
        .finally(() => { if (mcpBridge) mcpBridge.close().catch(() => {}); });
    },
  };
}
