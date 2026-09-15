/**
 * hmacAuth.js — Proves a Media Stream connection's tenant/prompt data
 * actually came from our Catalyst-hosted server, not an attacker who found
 * this service's public wss:// URL.
 *
 * Twilio doesn't sign the WebSocket upgrade the way it signs HTTP webhooks,
 * and this service deliberately has no Catalyst DataStore access (that's
 * the whole point of keeping it off Catalyst) — so it can't independently
 * check "is this accountSid a real tenant" the way the main app can.
 * Instead: the Catalyst server, which DOES have full tenant/DataStore
 * access at TwiML-generation time, signs (accountSid + timestamp) with a
 * secret only these two services share, and embeds the signature as another
 * <Parameter>. This service just verifies the signature and a freshness
 * window — it never needs to know what a "real" tenant looks like at all.
 */
import crypto from 'crypto';

const MAX_AGE_MS = 5 * 60 * 1000; // reject a replayed/stale TwiML payload

function getSecret() {
  const secret = process.env.MEDIA_STREAM_SHARED_SECRET;
  if (!secret) throw new Error('MEDIA_STREAM_SHARED_SECRET is not set');
  return secret;
}

/** Must produce the exact same signature as the Catalyst server's signer. */
export function sign(accountSid, timestamp) {
  return crypto.createHmac('sha256', getSecret()).update(`${accountSid}:${timestamp}`).digest('hex');
}

export function verify(accountSid, timestamp, signature) {
  if (!accountSid || !timestamp || !signature) return false;

  const age = Date.now() - Number(timestamp);
  if (!Number.isFinite(age) || age < 0 || age > MAX_AGE_MS) return false;

  const expected = Buffer.from(sign(accountSid, timestamp), 'hex');
  const actual   = Buffer.from(String(signature), 'hex');
  if (expected.length !== actual.length) return false;
  return crypto.timingSafeEqual(expected, actual); // constant-time compare
}
