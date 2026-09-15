/**
 * audioCodec.js — Audio format bridging between Twilio Media Streams and the
 * Gemini Live API. The two sides speak completely different formats:
 *
 *   Twilio  → 8kHz, 8-bit G.711 μ-law, base64, in 20ms (160-byte) frames
 *   Gemini  → 16-bit PCM in, 16kHz, base64 ("audio/pcm;rate=16000")
 *           ← 16-bit PCM out, 24kHz, base64 (fixed — not configurable)
 *
 * So every inbound caller frame needs μ-law → PCM16 decode + 8k → 16k
 * upsample before it can go to Gemini, and every Gemini response chunk needs
 * 24k → 8k downsample + PCM16 → μ-law encode before it can go back to
 * Twilio. This module is pure — no I/O, no knowledge of either transport —
 * so it can be unit-tested and reused independent of the WebSocket plumbing
 * in mediaStreamServer.js.
 *
 * Resampling is linear interpolation. That's a real quality compromise vs. a
 * proper polyphase/sinc resampler, but it's the standard, well-understood
 * tradeoff for telephony-grade (8kHz) voice — cheap, allocation-light, and
 * good enough that production Twilio↔LLM bridges commonly ship exactly this.
 */

// ── μ-law ⇄ linear PCM16 ──────────────────────────────────────────────────
// Standard ITU-T G.711 μ-law tables. Reference: the classic bias/segment
// algorithm (same one used in libsndfile, SoX, ffmpeg's g711.c, etc).

const MULAW_BIAS = 0x84;
const MULAW_CLIP = 32635;

function linearToMulawSample(sample) {
  let sign = (sample >> 8) & 0x80;
  if (sign !== 0) sample = -sample;
  if (sample > MULAW_CLIP) sample = MULAW_CLIP;
  sample += MULAW_BIAS;

  let exponent = 7;
  for (let mask = 0x4000; (sample & mask) === 0 && exponent > 0; mask >>= 1) {
    exponent--;
  }
  const mantissa = (sample >> (exponent + 3)) & 0x0f;
  const byte = ~(sign | (exponent << 4) | mantissa) & 0xff;
  return byte;
}

// Precomputed decode table — all 256 μ-law byte values → PCM16 sample.
const MULAW_DECODE_TABLE = new Int16Array(256);
(function buildDecodeTable() {
  for (let i = 0; i < 256; i++) {
    const byte = ~i & 0xff;
    const sign = byte & 0x80;
    const exponent = (byte >> 4) & 0x07;
    const mantissa = byte & 0x0f;
    let sample = ((mantissa << 3) + MULAW_BIAS) << exponent;
    sample -= MULAW_BIAS;
    MULAW_DECODE_TABLE[i] = sign !== 0 ? -sample : sample;
  }
})();

/** Buffer of μ-law bytes → Int16Array of PCM16 samples (same sample rate). */
export function mulawBufferToPcm16(mulawBuffer) {
  const out = new Int16Array(mulawBuffer.length);
  for (let i = 0; i < mulawBuffer.length; i++) {
    out[i] = MULAW_DECODE_TABLE[mulawBuffer[i]];
  }
  return out;
}

/** Int16Array of PCM16 samples → Buffer of μ-law bytes (same sample rate). */
export function pcm16ToMulawBuffer(pcm16) {
  const out = Buffer.alloc(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    out[i] = linearToMulawSample(pcm16[i]);
  }
  return out;
}

// ── Linear-interpolation resampling ───────────────────────────────────────

/** Int16Array at `fromRate` Hz → new Int16Array at `toRate` Hz. */
export function resamplePcm16(pcm16, fromRate, toRate) {
  if (fromRate === toRate) return pcm16;
  const ratio    = fromRate / toRate;
  const outLen   = Math.max(1, Math.round(pcm16.length / ratio));
  const out      = new Int16Array(outLen);
  const lastIdx  = pcm16.length - 1;

  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const idx0   = Math.min(lastIdx, Math.floor(srcPos));
    const idx1   = Math.min(lastIdx, idx0 + 1);
    const frac   = srcPos - idx0;
    out[i] = Math.round(pcm16[idx0] * (1 - frac) + pcm16[idx1] * frac);
  }
  return out;
}

// ── High-level direction-specific helpers ─────────────────────────────────

/**
 * Twilio inbound frame (base64 μ-law @ 8kHz) → base64 PCM16 @ 16kHz,
 * ready for Gemini's `sendRealtimeInput({ audio: { data, mimeType } })`.
 */
export function twilioPayloadToGeminiPcm16(base64Mulaw) {
  const mulawBuffer = Buffer.from(base64Mulaw, 'base64');
  const pcm8k        = mulawBufferToPcm16(mulawBuffer);
  const pcm16k        = resamplePcm16(pcm8k, 8000, 16000);
  return Buffer.from(pcm16k.buffer, pcm16k.byteOffset, pcm16k.byteLength).toString('base64');
}

/**
 * Gemini output chunk (base64 PCM16 @ 24kHz) → base64 μ-law @ 8kHz,
 * ready to send back to Twilio as a `media` event payload.
 */
export function geminiPcm16ToTwilioPayload(base64Pcm24k) {
  const pcmBuffer = Buffer.from(base64Pcm24k, 'base64');
  const pcm24k    = new Int16Array(pcmBuffer.buffer, pcmBuffer.byteOffset, pcmBuffer.length / 2);
  const pcm8k     = resamplePcm16(pcm24k, 24000, 8000);
  const mulaw     = pcm16ToMulawBuffer(pcm8k);
  return mulaw.toString('base64');
}

/** Splits a μ-law Buffer into 20ms (160-byte @ 8kHz) frames, base64-encoded,
 *  for pacing playback back to Twilio the way real call audio arrives. */
export function chunkMulawInto20msFrames(mulawBuffer) {
  const FRAME_BYTES = 160; // 8000 Hz * 0.02s * 1 byte/sample
  const frames = [];
  for (let offset = 0; offset < mulawBuffer.length; offset += FRAME_BYTES) {
    frames.push(mulawBuffer.subarray(offset, offset + FRAME_BYTES).toString('base64'));
  }
  return frames;
}
