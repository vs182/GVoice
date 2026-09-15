/**
 * Voice Hub — standalone Media Stream bridge (Render-hosted)
 *
 * The ONLY thing this service does is bridge Twilio Media Streams audio to
 * Gemini's Live API — see services/mediaBridge.js for why this had to move
 * off Catalyst. Everything else (webhooks, IVR, dashboard, DataStore) stays
 * on Catalyst untouched; index.js there points its `<Stream url>` at this
 * service's wss:// URL for the "ai-agent" IVR action only.
 *
 * Deliberately dependency-light: plain http (no Express) + ws. No Twilio
 * SDK, no Catalyst SDK, no DataStore access — this service can't and
 * shouldn't be able to look up tenant data itself; see hmacAuth.js for how
 * it trusts what Catalyst hands it instead.
 */

import http from 'http';
import { attachMediaStreamServer } from './services/mediaBridge.js';

const PORT = process.env.PORT || 3002;

const server = http.createServer((req, res) => {
  // Render's health check just needs any 200 — this also gives you a quick
  // manual "is it up" check by curling the service's URL directly.
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ status: 'ok', service: 'voice-hub-media-stream', timestamp: new Date().toISOString() }));
});

attachMediaStreamServer(server);

server.listen(PORT, () => {
  console.log(`Voice Hub media-stream bridge → port ${PORT}`);
  console.log(`  WebSocket endpoint: /media-stream`);
});
