"""
DeepCheck — optimized async pipeline
Target: 2-4s (text path) / 5-8s (video path)

Stack:
  - Google Fact Check API   → instant if already checked
  - Jina AI Reader          → text extraction without download
  - Apify TikTok Scraper    → captions/metadata
  - Tavily                  → source search (replaces Reddit+DDG+SerpAPI)
  - Gemini 2.5 Flash        → single merged call (not 3 sequential)
  - Redis                   → cache results, repeat URLs = <200ms
  - asyncio                 → everything runs in parallel
"""

import asyncio, json, os, re, time, uuid, tempfile, shutil, threading
import subprocess, urllib.parse, urllib.request
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from google import genai
from google.genai import types
import yt_dlp

# ── Optional fast deps (graceful fallback if not installed) ──────────────────
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import redis
    _redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    _rc = redis.from_url(_redis_url, decode_responses=True, socket_timeout=1)
    _rc.ping()
    HAS_REDIS = True
    print("[Cache] Redis connected")
except Exception:
    HAS_REDIS = False
    print("[Cache] Redis unavailable — caching disabled")

# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMNI_API_KEY")
TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY", "")
APIFY_API_KEY     = os.environ.get("APIFY_API_KEY", "")
GOOGLE_FC_API_KEY = os.environ.get("GOOGLE_FC_API_KEY", "")   # Fact Check Tools API
JINA_API_KEY      = os.environ.get("JINA_API_KEY", "")        # optional, works without key too
SERPAPI_KEY       = os.environ.get("SERPAPI_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

MODEL        = "gemini-2.5-flash"
MAX_DURATION = 180
CACHE_TTL    = 86400   # 24 hours
N_FRAMES     = 5       # frames to extract inline (no upload)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app  = Flask(__name__)
CORS(app)

JOBS      = {}
JOBS_LOCK = threading.Lock()

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_job():
    return {
        "status": "queued", "message": "Starting...", "progress_pct": 0,
        "error": None, "meta": None, "visual_analysis": None,
        "key_claims": None, "red_flags": None, "verification_points": None,
        "recommendation": None, "verdict": None, "confidence": None,
        "summary": None, "sources": None, "path": "unknown", "ms": 0,
    }

def job_set(task_id, **kw):
    with JOBS_LOCK:
        JOBS[task_id].update(kw)

def detect_platform(url):
    u = url.lower()
    if "tiktok.com"    in u: return "TikTok"
    if "instagram.com" in u: return "Instagram"
    if "youtube.com"   in u or "youtu.be" in u: return "YouTube"
    if "twitter.com"   in u or "x.com"    in u: return "X/Twitter"
    if "facebook.com"  in u: return "Facebook"
    return "Unknown"

def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",        "", text, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group() if m else text)

def cache_key(url):
    return f"deepcheck:v2:{url.strip().lower()}"

def cache_get(url):
    if not HAS_REDIS:
        return None
    try:
        raw = _rc.get(cache_key(url))
        return json.loads(raw) if raw else None
    except Exception:
        return None

def cache_set(url, data):
    if not HAS_REDIS:
        return
    try:
        _rc.setex(cache_key(url), CACHE_TTL, json.dumps(data))
    except Exception:
        pass

# ── Async HTTP helper ─────────────────────────────────────────────────────────

async def async_get(url, headers=None, timeout=6):
    """Simple async GET — uses aiohttp if available, falls back to sync."""
    if HAS_AIOHTTP:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.text()
    else:
        # fallback: run in thread pool so we don't block event loop
        loop = asyncio.get_event_loop()
        def _sync():
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "deepcheck/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        return await loop.run_in_executor(None, _sync)

async def async_post(url, payload, headers=None, timeout=10):
    if HAS_AIOHTTP:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.json()
    else:
        loop = asyncio.get_event_loop()
        def _sync():
            data = json.dumps(payload).encode()
            h = {"Content-Type": "application/json", **(headers or {})}
            req = urllib.request.Request(url, data=data, headers=h, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        return await loop.run_in_executor(None, _sync)

# ── Stage 0: Google Fact Check API ──────────────────────────────────────────

async def google_factcheck(query):
    """Returns existing fact-check result in ~500ms if one exists."""
    if not GOOGLE_FC_API_KEY:
        return None
    try:
        q = urllib.parse.quote_plus(query[:100])
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={q}&key={GOOGLE_FC_API_KEY}&pageSize=3"
        raw = await async_get(url, timeout=4)
        data = json.loads(raw)
        claims = data.get("claims", [])
        if not claims:
            return None
        # Pull best result
        c = claims[0]
        review = (c.get("claimReview") or [{}])[0]
        rating = review.get("textualRating", "")
        publisher = review.get("publisher", {}).get("name", "")
        url_link = review.get("url", "")
        print(f"[FactCheck API] Found: {rating} by {publisher}")
        return {
            "rating": rating,
            "publisher": publisher,
            "url": url_link,
            "claim_text": c.get("text", ""),
        }
    except Exception as e:
        print(f"[FactCheck API] Failed: {e}")
        return None

# ── Stage 1a: Jina AI text extraction ────────────────────────────────────────

async def jina_extract(url):
    """Extract text/captions from URL without downloading video. ~1s."""
    try:
        headers = {"Accept": "application/json"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        jina_url = f"https://r.jina.ai/{url}"
        raw = await async_get(jina_url, headers=headers, timeout=8)
        # Jina returns markdown — grab meaningful text
        text = raw.strip()
        if len(text) > 100:
            print(f"[Jina] Got {len(text)} chars")
            return text[:3000]
        return None
    except Exception as e:
        print(f"[Jina] Failed: {e}")
        return None

# ── Stage 1b: Apify TikTok scraper ───────────────────────────────────────────

async def apify_scrape(url):
    """Get TikTok/Instagram metadata + captions via Apify. ~1-2s."""
    if not APIFY_API_KEY:
        return None
    platform = detect_platform(url)
    # Pick actor per platform
    actor = {
        "TikTok":    "clockworks~free-tiktok-scraper",
        "Instagram": "apify~instagram-post-scraper",
    }.get(platform)
    if not actor:
        return None
    try:
        endpoint = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
        payload  = {"postURLs": [url], "resultsPerPage": 1} if platform == "TikTok" \
                   else {"directUrls": [url], "resultsLimit": 1}
        params   = f"?token={APIFY_API_KEY}&timeout=15&memory=256"
        data = await async_post(endpoint + params, payload, timeout=20)
        if data and isinstance(data, list) and data[0]:
            item = data[0]
            text = " ".join(filter(None, [
                item.get("text") or item.get("description") or "",
                item.get("videoMeta", {}).get("subtitleLinks", [""])[0] if platform == "TikTok" else "",
                " ".join(item.get("hashtags", [])),
            ]))
            print(f"[Apify] Got metadata, text len: {len(text)}")
            return {
                "text":      text[:2000],
                "title":     item.get("text", "")[:200],
                "uploader":  item.get("authorMeta", {}).get("name", "") or item.get("ownerUsername", ""),
                "likes":     item.get("diggCount") or item.get("likesCount"),
                "views":     item.get("playCount") or item.get("videoPlayCount"),
                "comments":  item.get("commentCount") or item.get("commentsCount"),
            }
    except Exception as e:
        print(f"[Apify] Failed: {e}")
    return None

# ── Stage 1c: yt-dlp metadata only (no download) ─────────────────────────────

async def ytdlp_metadata(url):
    """Extract metadata + description without downloading. ~2s."""
    loop = asyncio.get_event_loop()
    def _extract():
        opts = {
            "quiet": True, "no_warnings": True,
            "skip_download": True,           # KEY: no video download
            "socket_timeout": 10,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title":       info.get("title", ""),
            "description": (info.get("description") or "")[:1000],
            "uploader":    info.get("uploader") or info.get("channel", ""),
            "duration":    info.get("duration", 0),
            "platform":    detect_platform(url),
            "view_count":  info.get("view_count"),
            "like_count":  info.get("like_count"),
            "thumbnail":   info.get("thumbnail", ""),
            # automatic captions
            "captions":    _get_captions(info),
        }
    try:
        return await loop.run_in_executor(None, _extract)
    except Exception as e:
        print(f"[yt-dlp meta] Failed: {e}")
        return None

def _get_captions(info):
    """Pull auto-generated captions text from yt-dlp info."""
    try:
        subs = info.get("automatic_captions") or info.get("subtitles") or {}
        for lang in ["en", "en-US", list(subs.keys())[0] if subs else None]:
            if not lang or lang not in subs:
                continue
            entries = subs[lang]
            # entries is list of {url, ext} — grab first json3 or vtt
            for e in entries:
                if e.get("ext") in ("json3", "vtt", "srv3"):
                    try:
                        req = urllib.request.Request(e["url"], headers={"User-Agent": "deepcheck/2.0"})
                        with urllib.request.urlopen(req, timeout=5) as r:
                            raw = r.read().decode("utf-8", errors="replace")
                        # Strip tags/timestamps, return plain text
                        text = re.sub(r"<[^>]+>", " ", raw)
                        text = re.sub(r"\d{2}:\d{2}:\d{2}[^\n]*\n", "", text)
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) > 50:
                            return text[:2000]
                    except Exception:
                        pass
    except Exception:
        pass
    return ""

# ── Stage 1d: inline frame extraction (no Gemini upload) ────────────────────

async def extract_frames_inline(url, n=N_FRAMES):
    """
    Get stream URL via yt-dlp, pipe frames directly via ffmpeg.
    Returns list of base64 JPEG strings. No file upload needed.
    """
    loop = asyncio.get_event_loop()

    def _get_stream_url():
        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "format": "best[height<=360][ext=mp4]/best[height<=360]/worst",
                "socket_timeout": 8}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("url") or info.get("manifest_url"), info.get("duration", 30)

    try:
        stream_url, duration = await asyncio.wait_for(
            loop.run_in_executor(None, _get_stream_url), timeout=10
        )
    except Exception as e:
        print(f"[Frames] Stream URL failed: {e}")
        return []

    if not stream_url:
        return []

    # Calculate timestamps spread across video
    duration = min(duration or 30, 60)
    timestamps = [duration * i / (n + 1) for i in range(1, n + 1)]

    frames = []
    for ts in timestamps:
        try:
            cmd = [
                "ffmpeg", "-ss", str(ts),
                "-i", stream_url,
                "-frames:v", "1",
                "-vf", "scale=640:-1",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-q:v", "5",
                "pipe:1",
            ]
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: subprocess.run(
                    cmd, capture_output=True, timeout=8
                )),
                timeout=10
            )
            if result.returncode == 0 and result.stdout:
                import base64
                frames.append(base64.b64encode(result.stdout).decode())
        except Exception as e:
            print(f"[Frames] Frame at {ts:.1f}s failed: {e}")

    print(f"[Frames] Got {len(frames)}/{n} frames inline")
    return frames

# ── Stage 2: Tavily source search ────────────────────────────────────────────

async def tavily_search(query):
    """Fast source search purpose-built for AI fact-checking. ~1s."""
    if not TAVILY_API_KEY:
        return []
    try:
        data = await async_post(
            "https://api.tavily.com/search",
            {
                "api_key":      TAVILY_API_KEY,
                "query":        query[:300],
                "search_depth": "basic",
                "max_results":  5,
                "include_answer": False,
            },
            timeout=6
        )
        results = data.get("results", [])
        return [
            {"name": r.get("title", ""), "url": r.get("url", ""), "description": r.get("content", "")[:200]}
            for r in results if r.get("url")
        ][:5]
    except Exception as e:
        print(f"[Tavily] Failed: {e}")
        return []

# ── Stage 3: Single merged Gemini call ───────────────────────────────────────

def gemini_analyze(text_context, frames, meta, fc_result):
    """
    One Gemini call that does what your 3 sequential calls did.
    Text path: ~1-2s. Frame path: ~3-5s.
    """
    parts = []

    # Add frames if available (visual path)
    for b64 in frames:
        parts.append(types.Part(inline_data=types.Blob(
            mime_type="image/jpeg",
            data=b64
        )))

    # Build unified prompt
    fc_hint = ""
    if fc_result:
        fc_hint = f"\nEXISTING FACT-CHECK: {fc_result['publisher']} rated this '{fc_result['rating']}' — claim: \"{fc_result['claim_text']}\"\n"

    prompt = f"""You are a senior fact-checker. Analyze this social media content and return a verdict.

PLATFORM: {meta.get('platform','Unknown')}
TITLE: {meta.get('title','')}
UPLOADER: {meta.get('uploader','')}
DESCRIPTION: {meta.get('description','')[:400]}
CAPTIONS/TEXT: {text_context[:1500] if text_context else 'None extracted'}
{fc_hint}
{"VISUAL FRAMES: " + str(len(frames)) + " frames provided above." if frames else "NOTE: No video frames — analyze from text only."}

Return ONLY valid JSON, no markdown:
{{
  "verdict": "LIKELY AUTHENTIC|NEEDS CONTEXT|POSSIBLY MISLEADING|LIKELY FAKE|CANNOT DETERMINE",
  "confidence": 75,
  "summary": "2-3 sentences covering what the content claims, what's verified, and why this verdict.",
  "visual_analysis": "What the frames show (or 'text-only analysis' if no frames).",
  "key_claims": ["Claim 1", "Claim 2", "Claim 3"],
  "red_flags": ["Only real manipulation or false claims — not just sensitivity"],
  "verification_points": ["Specific checkable fact: how to verify it"],
  "recommendation": "Practical advice for the viewer."
}}

VERDICT RULES:
- Real footage + real claims = LIKELY AUTHENTIC (72-88%)
- Real footage + misleading framing = NEEDS CONTEXT (max 65%)
- Specific false factual claims = POSSIBLY MISLEADING (max 55%)
- Fabricated/AI-generated content = LIKELY FAKE (max 88%)
- Insufficient info = CANNOT DETERMINE (30-55%)
- Sensitivity alone is NOT a red flag
- News org branding = strong positive signal"""

    parts.append(types.Part(text=prompt))

    resp = gemini_client.models.generate_content(
        model=MODEL,
        contents=[types.Content(parts=parts)],
    )
    return parse_json(resp.text.strip())

# ── Main async pipeline ───────────────────────────────────────────────────────

async def fast_pipeline(task_id, url, context):
    t0 = time.time()
    platform = detect_platform(url)

    # ── Check cache first ──────────────────────────────────────────────────
    cached = cache_get(url)
    if cached:
        print(f"[Cache] HIT — returning in {(time.time()-t0)*1000:.0f}ms")
        job_set(task_id, **cached, status="done", message="Done (cached).",
                progress_pct=100, ms=int((time.time()-t0)*1000))
        return

    job_set(task_id, status="analyzing", message="Extracting content...", progress_pct=10)

    # ── Phase 1: parallel text extraction + fact-check lookup ─────────────
    # Run everything that doesn't need video simultaneously
    text_tasks = await asyncio.gather(
        jina_extract(url),
        apify_scrape(url) if platform in ("TikTok", "Instagram") else asyncio.sleep(0),
        ytdlp_metadata(url),
        return_exceptions=True
    )

    jina_text  = text_tasks[0] if not isinstance(text_tasks[0], Exception) else None
    apify_data = text_tasks[1] if not isinstance(text_tasks[1], Exception) else None
    meta       = text_tasks[2] if not isinstance(text_tasks[2], Exception) else {}

    if not meta:
        meta = {}
    meta["platform"]      = platform
    meta["user_context"]  = context

    # Merge all text sources
    text_parts = []
    if jina_text:                        text_parts.append(f"[Jina]\n{jina_text}")
    if apify_data and apify_data.get("text"): text_parts.append(f"[Apify]\n{apify_data['text']}")
    if meta.get("captions"):             text_parts.append(f"[Captions]\n{meta['captions']}")
    if meta.get("description"):          text_parts.append(f"[Description]\n{meta['description']}")
    if context:                          text_parts.append(f"[User context]\n{context}")
    text_context = "\n\n".join(text_parts)

    has_text = len(text_context.strip()) > 100
    print(f"[Pipeline] Text extracted: {len(text_context)} chars, has_text={has_text}, t={time.time()-t0:.1f}s")

    job_set(task_id, message="Fact-checking...", progress_pct=30, meta=meta)

    # ── Phase 2: parallel — Gemini analysis + source search ───────────────
    # Build search query from what we have so far
    search_query = (meta.get("title") or "") + " " + " ".join(text_context.split()[:20])
    search_query = search_query.strip()[:200]

    frames = []
    if not has_text:
        # Need frames — extract while we start other tasks
        job_set(task_id, message="Extracting video frames...", progress_pct=35)
        frames = await extract_frames_inline(url, n=N_FRAMES)
        if not frames:
            job_set(task_id, message="Downloading video (fallback)...", progress_pct=40)
            # Last resort: download small video
            frames = await download_and_frame_fallback(url)

    job_set(task_id, message="Analyzing with AI...", progress_pct=50)

    # Fire Gemini + sources in parallel
    loop = asyncio.get_event_loop()

    gemini_task = loop.run_in_executor(
        None, gemini_analyze, text_context, frames, meta, None
    )
    fc_task     = google_factcheck(search_query)
    tavily_task = tavily_search(search_query)

    results = await asyncio.gather(gemini_task, fc_task, tavily_task, return_exceptions=True)

    analysis   = results[0] if not isinstance(results[0], Exception) else {}
    fc_result  = results[1] if not isinstance(results[1], Exception) else None
    sources    = results[2] if not isinstance(results[2], Exception) else []

    if isinstance(results[0], Exception):
        print(f"[Gemini] Failed: {results[0]}")
        raise results[0]

    # ── If Google Fact Check API found something, factor it in ────────────
    if fc_result:
        job_set(task_id, message="Cross-checking existing fact-checks...", progress_pct=85)
        # Quick re-call with fact-check context (cheap text-only call)
        try:
            analysis = await loop.run_in_executor(
                None, gemini_analyze, text_context + f"\n\nEXISTING FACT-CHECK: {fc_result}", [], meta, fc_result
            )
        except Exception as e:
            print(f"[FC merge] Failed: {e}")

    elapsed = int((time.time() - t0) * 1000)
    path    = "text" if has_text else ("frames" if frames else "fallback")
    print(f"[Pipeline] Done in {elapsed}ms via {path} path")

    final = {
        "verdict":            analysis.get("verdict", "CANNOT DETERMINE"),
        "confidence":         analysis.get("confidence", 40),
        "summary":            analysis.get("summary", ""),
        "visual_analysis":    analysis.get("visual_analysis", ""),
        "key_claims":         analysis.get("key_claims", []),
        "red_flags":          analysis.get("red_flags", []),
        "verification_points":analysis.get("verification_points", []),
        "recommendation":     analysis.get("recommendation", ""),
        "sources":            sources,
        "path":               path,
        "ms":                 elapsed,
    }

    cache_set(url, final)

    job_set(task_id,
        status="done", message="Done.", progress_pct=100, **final
    )


async def download_and_frame_fallback(url):
    """Last resort: download small video, extract frames inline, no Gemini upload."""
    loop = asyncio.get_event_loop()
    tmp  = tempfile.mkdtemp()
    try:
        def _download():
            opts = {
                "quiet": True, "no_warnings": True,
                "format": "worst[ext=mp4]/worst",
                "outtmpl": os.path.join(tmp, "video.%(ext)s"),
                "socket_timeout": 10,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info), info.get("duration", 30)

        video_path, duration = await asyncio.wait_for(
            loop.run_in_executor(None, _download), timeout=25
        )

        if not os.path.exists(video_path):
            video_path = os.path.join(tmp, "video.mp4")

        duration = min(duration or 30, 60)
        timestamps = [duration * i / (N_FRAMES + 1) for i in range(1, N_FRAMES + 1)]
        frames = []
        import base64
        for ts in timestamps:
            try:
                cmd = [
                    "ffmpeg", "-ss", str(ts), "-i", video_path,
                    "-frames:v", "1", "-vf", "scale=640:-1",
                    "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "pipe:1",
                ]
                r = subprocess.run(cmd, capture_output=True, timeout=6)
                if r.returncode == 0 and r.stdout:
                    frames.append(base64.b64encode(r.stdout).decode())
            except Exception:
                pass
        print(f"[Fallback] Got {len(frames)} frames")
        return frames
    except Exception as e:
        print(f"[Fallback] Failed: {e}")
        return []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Background thread (bridges sync Flask → async pipeline) ──────────────────

def background_process(task_id, url, context):
    try:
        asyncio.run(fast_pipeline(task_id, url, context))
    except ValueError as e:
        job_set(task_id, status="error", error=str(e))
    except Exception as e:
        job_set(task_id, status="error", error=f"Analysis failed: {str(e)[:200]}")

# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    body    = request.json or {}
    url     = body.get("url", "").strip()
    context = body.get("context", "").strip()
    if not url:
        return jsonify({"error": "Please provide a video URL"}), 400

    # Instant cache hit — don't even spin up a thread
    cached = cache_get(url)
    if cached:
        task_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[task_id] = {**make_job(), **cached,
                             "status": "done", "message": "Done (cached).",
                             "progress_pct": 100}
        return jsonify({"task_id": task_id, "cached": True})

    task_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[task_id] = make_job()
    threading.Thread(target=background_process, args=(task_id, url, context), daemon=True).start()
    return jsonify({"task_id": task_id, "cached": False})

@app.route("/status/<task_id>")
def status(task_id):
    with JOBS_LOCK:
        job = JOBS.get(task_id)
        if not job:
            return jsonify({"error": "Unknown task_id"}), 404
        return jsonify(job)

@app.route("/sw.js")
def service_worker():
    return Response("""
const CACHE='deepcheck-v13';
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/']))));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));});
""", mimetype="application/javascript")

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("RENDER") is None
    app.run(host="0.0.0.0", port=port, debug=debug)