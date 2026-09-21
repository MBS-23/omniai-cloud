# Deployment

## Where each profile can run

The **`paper` profile** runs the paper's own models in-process. It needs about
8 GB of memory with every model loaded, roughly 10 GB of disk for the image,
and no API key. It fits any Linux server or VM with that much memory; the
free one used here is an Oracle Cloud Ampere A1 instance. It does not fit
serverless platforms (Vercel, Netlify, Cloudflare Workers), whose bundles are
capped far below the size of the weights.

The **`lite` profile** carries no weights and calls a hosted API instead. It is
the only profile that fits a serverless function, and it is bound by that API's
rate limits.

## Oracle Cloud: the paper profile, live, free

Oracle's Always Free tier includes Ampere A1 capacity of 4 cores and 24 GB of
memory, enough for every model with room to spare. PyTorch, PaddlePaddle,
OpenCV and spaCy all publish ARM64 builds, so the same Dockerfile builds there
natively.

1. **Create the account** at [signup.cloud.oracle.com](https://signup.cloud.oracle.com).
   Oracle verifies a card but does not charge Always Free resources.
2. **Create the instance**: Compute, Instances, Create instance.
   - Image: Canonical Ubuntu 24.04 (the aarch64 build is chosen automatically
     for the shape below).
   - Shape: `VM.Standard.A1.Flex`, 4 OCPUs, 24 GB memory.
   - Boot volume: 100 GB.
   - SSH keys: paste the public half of your deploy key.
   If the console reports "out of capacity", pick another availability domain
   or try again later; free A1 capacity is released continuously.
3. **Open HTTP and HTTPS**: Networking, the instance's virtual cloud network,
   its security list, Add ingress rules: source `0.0.0.0/0`, TCP, destination
   ports `80` and `443`.
4. **Deploy** from your machine, with the instance's public IP:

   ```bash
   bash scripts/deploy_oracle.sh 203.0.113.9 ~/.ssh/omniai_oracle
   ```

   It sends the committed code, installs updates, Docker and the firewall
   rules, builds the image on the server (the first build downloads about
   6 GB of models) and starts it behind Caddy, which obtains an HTTPS
   certificate automatically. The site is served at
   `https://203-0-113-9.sslip.io`: sslip.io is a public DNS name that points at
   the IP, so no domain has to be bought. Run the same command to deploy an
   update.

To use your own domain instead, point it at the IP and pass it:
`SITE_ADDRESS=omniai.example.org bash scripts/deploy_oracle.sh 203.0.113.9`.

## Hugging Face Spaces (needs a PRO subscription)

The repository's `Dockerfile` is a Space-ready image: it listens on 7860, runs
as uid 1000, fetches every model from the Hub while building, proves each one
loads with the network off, and pre-loads them all at startup.

Hugging Face now reserves Docker Spaces on its CPU hardware for PRO accounts;
creating one on a free account is refused with `402 Payment Required`. With
PRO, the steps below publish the same image.

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
| Oracle Cloud Always Free (Ampere A1) | **paper** | 4 cores and 24 GB at no cost. See above. |
| Hugging Face Spaces | paper | Docker Spaces need a PRO subscription. |
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

## Scaling

Run one worker per container. Each worker loads every model once, so a second
worker in the same container doubles the memory for no gain on a CPU-bound
host; `gunicorn.conf.py` sets one worker with four threads. To serve more
people, run more containers behind a load balancer.

If the app sits behind reverse proxies, set `OMNIAI_TRUSTED_PROXIES` to how
many there are (a Hugging Face Space is detected and set to one). Leave it at
`0` when the app is reached directly.

## Operating it

Health: `GET /api/health` returns `200` with `{"status": "ok"}` when the app
can serve requests, and `503` when it cannot. On a local profile it also
reports how many models are in memory and whether start-up loading is still
running. When something is misconfigured, the reason is written to the log,
not to the response.

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
