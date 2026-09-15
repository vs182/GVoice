# Voice Hub — Media Stream Bridge

Standalone WebSocket service: bridges Twilio Media Streams audio to Gemini's
Live API for the "AI Voice Agent" IVR action.

Lives outside the main Catalyst-hosted app (`twilio-catalyst-server`)
because Zoho Catalyst's gateway doesn't pass WebSocket upgrades through to
AppSail — confirmed by a live test, not assumed. Everything else (all other
webhooks, the dashboard, DataStore) stays on Catalyst untouched; this
service only handles `/media-stream`.

## How it's trusted

This service has **no Catalyst DataStore access and no Twilio credentials**
— on purpose. It can't independently verify "is this accountSid a real
tenant." Instead, `twilio-catalyst-server` (which has full tenant context
when it generates the call's TwiML) signs `accountSid + timestamp` with a
shared secret and passes it via `<Parameter>` elements on the `<Stream>`
tag. This service just checks that signature (see `services/hmacAuth.js`)
before opening a Gemini session.

## Deploying on Render

1. Push this directory to its own GitHub repo.
2. In Render: **New → Web Service**, connect that repo.
3. Build command: `npm install`. Start command: `npm start`.
4. Environment variables (Render dashboard → Environment):
   - `GEMINI_API_KEY` — same key used elsewhere in Voice Hub
   - `MEDIA_STREAM_SHARED_SECRET` — must exactly match `twilio-catalyst-server`'s value
   - `GEMINI_MODEL` (optional, defaults to `gemini-3.1-flash-live-preview`)
   - `GEMINI_VOICE` (optional, defaults to `Puck`)
5. Deploy. Render gives you a URL like `https://<app>.onrender.com`.
6. Set `MEDIA_STREAM_URL=wss://<app>.onrender.com/media-stream` in
   `twilio-catalyst-server`'s `app-config.json` and redeploy that server.

## Local dev

```bash
npm install
GEMINI_API_KEY=... MEDIA_STREAM_SHARED_SECRET=... npm run dev
```

Health check: `GET /` returns `{"status":"ok",...}`. The actual endpoint
Twilio connects to is `wss://<host>/media-stream`.
