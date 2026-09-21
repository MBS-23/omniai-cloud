# Deployment

## What can and cannot be deployed

Only the **`lite` profile** is deployable to ordinary hosting. It has no PyTorch
dependency and no model weights, so the image is small and starts in about a
second.

The **`full` profile carries roughly 6 GB of weights** and wants a GPU. It does
not fit free or hobby tiers, and it does not fit serverless platforms at all —
not Vercel, not Netlify, not Cloudflare Workers, all of which cap bundles far
below this and impose execution limits shorter than a BLIP inference. Run `full`
locally, or on a GPU VM you control.

## Before you deploy

1. **Rotate the API key.** Generate a fresh one at
   [console.mistral.ai](https://console.mistral.ai/api-keys). Set it as a
   platform secret, never in the repository.
2. **Confirm `.env` is ignored.** `git check-ignore .env` should print `.env`.
3. **Set `OMNIAI_DEBUG=false`.** It defaults to false; confirm it anyway.
4. **Set a rate limit** you are willing to pay for. Every request in `lite` is a
   billed API call. `OMNIAI_RATE_LIMIT_PER_MINUTE=20` is the default.
5. **Set `OMNIAI_MAX_UPLOAD_BYTES`** to the smallest value that serves your use.
6. **Run the tests.** `python -m pytest`

## Docker

```bash
docker compose up --build
```

Then check it is healthy:

```bash
curl -fsS http://localhost:8000/api/health
```

The compose file runs the container read-only, drops all capabilities, sets
`no-new-privileges`, and mounts `/app/temp_uploads` as a 128 MB tmpfs so uploads
never touch the container filesystem.

## Platform notes

| Platform | Fit | Notes |
|---|---|---|
| Fly.io | good | `fly launch` reads the Dockerfile. Smallest paid VM is enough. |
| Render | good | Web Service from the Dockerfile. Free tier sleeps; first request is slow. |
| Railway | good | Detects the Dockerfile. Set the secret in the dashboard. |
| Hugging Face Spaces | good | Docker Space. Set the key as a Space secret, not in the repo. |
| Google Cloud Run | good | Needs `--min-instances 1` if you want the rate limiter to hold. |
| A VM with nginx | good | See the proxy section below. |
| Vercel | **yes, lite + Gemini** | The configured target. See *Vercel* below. The offline profile cannot run here: no disk for 10 GB of weights. |

### Vercel

The repository is already configured: `pyproject.toml` points Vercel at
`backend.main:app`, `requirements.txt` installs only the lite profile (about
57 MB, no PyTorch), and `vercel.json` sets a 60-second function limit.

On Vercel the app detects that it is serverless, defaults to the `lite`
profile, and writes uploads under `/tmp`. The only thing it needs from you:

| Variable | Value |
|---|---|
| `OMNIAI_GEMINI_API_KEY` | a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

Set it in the Vercel dashboard under **Project → Settings → Environment
Variables**, for Production and Preview, then redeploy. Never commit it.

To deploy: open [vercel.com/new](https://vercel.com/new), import the
repository, add the variable above before the first deploy, and deploy.
Every later push to `main` redeploys automatically.

Progress still streams, but the whole analysis must finish inside the
60-second function limit. Image analysis with the ensemble makes three or four
model calls, which fits comfortably on Gemini's latency.

### Fly.io

```bash
fly launch --no-deploy
```

```bash
fly secrets set OMNIAI_MISTRAL_API_KEY=your-key-here
```

```bash
fly deploy
```

### Behind nginx

Server-Sent Events need buffering disabled, or progress arrives all at once at
the end:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Required for /api/analyze/*/stream
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
}
```

The application already sends `X-Accel-Buffering: no` on stream responses, which
nginx honours; the explicit directives are belt and braces.

## Scaling, and one real caveat

**The rate limiter is per process.** It is an in-memory sliding window. Two
gunicorn workers means a client can make twice the configured number of
requests. The Dockerfile therefore runs **one worker with eight threads**, which
is the right shape anyway: in `lite` every request is waiting on an upstream
API, so threads are cheap and a second process buys nothing.

If you need more than one worker or more than one replica, move the limiter to
shared state before you do. `SlidingWindowRateLimiter` in
`backend/utils/security.py` is a single class with one method; backing it with
Redis is a small change. Until then, a platform-level rate limit is a better
control than adding workers.

**`X-Forwarded-For` is trusted** for client identification when present. That is
correct behind a proxy you control and wrong if the app is directly exposed,
where a client can forge the header to evade the limit. Put it behind a proxy,
or drop that branch in `client_identifier`.

## Operating it

Health: `GET /api/health` returns `200` when a backend can serve requests and
`503` when it cannot — an invalid or missing key shows as `503` with
`status: degraded`. It reports the profile, the capability list and the upload
limit. It never reports the key.

Logs go to stdout, and to `logs/omniai.log` when that directory is writable. In
a read-only container only stdout is used, which is what platform log collectors
expect.

Uploads are deleted in a `finally` block on every request. `purge_stale_uploads`
runs at startup and removes anything older than an hour, which covers a process
killed mid-request.

## Cost

Every `lite` request is billed by Mistral. A text analysis is one or two calls;
an image analysis is three or four; a scanned PDF is one OCR call per page up to
30 pages. The rate limit is the main cost control — set it deliberately, and
consider putting the deployment behind authentication if the URL is public.
