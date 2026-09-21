# Deployment

## Where each profile can run

The **`paper` profile** runs the paper's own models in-process. It needs about
8 GB of memory with every model loaded and roughly 9 GB of disk for the image,
and no API key. That fits a **Hugging Face Space** on the free CPU tier (2 vCPU,
16 GB), or any Docker host with 8 GB or more. It does not fit serverless
platforms (Vercel, Netlify, Cloudflare Workers): their bundles are capped far
below the size of the weights.

The **`lite` profile** carries no weights and calls a hosted API instead. It is
the only profile that fits a serverless function, and it is bound by that API's
rate limits.

## Hugging Face Spaces: the paper profile, live

The repository's `Dockerfile` is a Space-ready image: it listens on 7860, runs
as uid 1000, fetches every model from the Hub while building, proves each one
loads with the network off, and pre-loads them all at startup.

1. Log in once, with a token that has **write** access
   (create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   hf auth login
   ```

2. Publish:

   ```bash
   python scripts/deploy_space.py YOUR-USERNAME/omniai-cloud
   ```

   The script creates a public Docker Space if it does not exist, uploads only
   what the image needs, and uses `deploy/huggingface/README.md` as the Space
   card. Run it again to deploy an update.

3. The first build takes about 15 minutes (packages, then about 6 GB of
   weights). Follow it on the Space's **Logs** tab. When it shows
   *Running*, the app is at `https://YOUR-USERNAME-omniai-cloud.hf.space`.

Free Spaces sleep after 48 hours without visitors and wake on the next visit;
waking takes about a minute while the models load. `/api/health` reports
`"preloading": true` until they are all in memory. Requests that arrive during
that minute are served as soon as the model they need has loaded.

The app refuses to be framed by any site except `huggingface.co`, and only when
it detects it is running on a Space (the `SPACE_ID` variable Spaces set).

## Before you deploy

1. **Confirm `.env` is ignored.** `git check-ignore .env` should print `.env`.
2. **Set `OMNIAI_DEBUG=false`.** It defaults to false; confirm it anyway.
3. **Set a rate limit.** On `paper` it protects a shared CPU from one visitor;
   on `lite` it also protects your API quota. `OMNIAI_RATE_LIMIT_PER_MINUTE=20`
   is the default.
4. **Set `OMNIAI_MAX_UPLOAD_BYTES`** to the smallest value that serves your use.
5. **For `lite` only: use a fresh API key**, set as a platform secret. A key
   that has ever been pasted into a chat, an issue or a commit should be
   replaced.
6. **Run the tests.** `python -m pytest`

## Docker, anywhere

```bash
docker compose up --build
```

Then check it is healthy:

```bash
curl -fsS http://localhost:7860/api/health
```

The compose file drops all capabilities, sets `no-new-privileges`, and mounts
`/app/temp_uploads` as a tmpfs so uploads never touch the container
filesystem. It sets `OMNIAI_PRELOAD=none` so the image also runs on a laptop
with less than 8 GB: models then load on first use. Set it to `all` on a host
with the memory.

## Platform notes

| Platform | Profile | Notes |
|---|---|---|
| Hugging Face Spaces | **paper** | Free CPU tier is enough. See above. |
| A VM or dedicated server | paper | 8 GB of memory or more. See the proxy section below. |
| Google Cloud Run | paper | 8 GiB memory, `--min-instances 1` if the rate limiter should hold. |
| Fly.io, Render, Railway | paper on an 8 GB plan, or lite | Their free tiers are too small for the paper's models. |
| Vercel | lite | Serverless: no room for the weights. See *Vercel* below. |

### Vercel

The repository is already configured: `pyproject.toml` points Vercel at
`backend.main:app`, `requirements.txt` installs only the lite profile (about
57 MB, no PyTorch), and `vercel.json` sets a 60-second function limit.

On Vercel the app detects that it is serverless, defaults to the `lite`
profile, and writes uploads under `/tmp`. The only thing it needs from you:

| Variable | Value |
|---|---|
| `OMNIAI_GEMINI_API_KEY` | a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are accepted as alternative names.

1. Open the project in the Vercel dashboard, then **Settings → Environment
   Variables**.
2. **Key** `OMNIAI_GEMINI_API_KEY`, **Value** the key itself, with no quotes
   and no spaces.
3. Under **Environments**, tick **Production** (the live URL). Tick Preview and
   Development too if you use them.
4. **Save**.
5. **Redeploy.** Vercel reads variables when a deployment is built, so a
   running deployment never sees a variable added after it. Go to
   **Deployments**, open the latest, **⋯ → Redeploy**, or push any commit to
   `main`.
6. Open `/api/health` on the live URL. It should say `"status": "ok"` and
   `"hosted_provider": "gemini"`.

Never commit the key.

To deploy from scratch: open [vercel.com/new](https://vercel.com/new), import
the repository, add the variable above before the first deploy, and deploy.
Every later push to `main` redeploys automatically.

#### When the live site says "degraded"

While no key is configured, `/api/health` answers `503` and includes a
`diagnostics` block. It lists variable **names** only, never values:

| What `/api/health` shows | What it means | Fix |
|---|---|---|
| a name under `set_but_empty` | the variable exists in this environment, but its value is blank | edit it, paste the key, save, redeploy |
| a name under `similar_names_found` | a variable with a near-miss name, such as `GEMINI_KEY` | rename it to `OMNIAI_GEMINI_API_KEY`, redeploy |
| both lists empty | no key variable reaches this deployment at all | the variable was saved for Preview or Development only, saved on a different Vercel project, or added after this deployment was built. Tick Production and redeploy |
| `status: ok`, but every analysis fails | the key is present but Google rejects it (revoked, mistyped, or its project lacks access) | the result panel quotes Gemini's error; create a new key at aistudio.google.com and replace the value |

Progress still streams, but the whole analysis must finish inside the
60-second function limit. Image analysis with the ensemble makes three or four
model calls, which fits comfortably on Gemini's latency.

### Fly.io

```bash
fly launch --no-deploy
```

```bash
fly scale memory 8192
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
`503` when it cannot. A missing key shows as `503` with `status: degraded`, a
`detail` saying what to set and where, and the `diagnostics` block described
under *Vercel*. It reports the profile, the provider, the capability list and
the upload limit. It never reports a key.

Logs go to stdout, and to `logs/omniai.log` when that directory is writable. In
a read-only container only stdout is used, which is what platform log collectors
expect.

Uploads are deleted in a `finally` block on every request. `purge_stale_uploads`
runs at startup and removes anything older than an hour, which covers a process
killed mid-request.

## Cost

Every `lite` request is a call against your provider's quota. A text analysis
is one or two calls; an image analysis is three or four with the ensemble; a
scanned PDF is one call for the whole document on Gemini, or one OCR call per
page (up to 30) on Mistral. Gemini's free tier covers a demo; past that, usage
is billed to the Google Cloud project behind the key. The rate limit is the
main cost control. Set it deliberately, and consider putting the deployment
behind authentication if the URL is public.
