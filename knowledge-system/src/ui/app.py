"""
Chat UI for the Credit Union Knowledge System.
Run: cd knowledge-system/src/ui && python app.py
Then open http://localhost:5000 in your browser.
"""
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import date
from pathlib import Path

from flask import Flask, request, jsonify, Response

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
sys.path.insert(0, str(SRC / "harness"))

import importlib.util
from contract import Snapshot, Page

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
ks = None
_system_lock = threading.Lock()
logger = logging.getLogger(__name__)


def load_system():
    """Load the knowledge base once, publishing it only after a full ingest."""
    global ks
    if ks is not None:
        return ks
    spec = importlib.util.spec_from_file_location("solution", HERE.parent / "solution.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the knowledge system")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidate = mod.build()

    root = SRC / "snapshots"
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()):
        manifest = json.loads((d / "manifest.json").read_text())
        pages = [
            Page(
                url=p["url"],
                title=p.get("title", ""),
                markdown=(d / p["file"]).read_text(encoding="utf-8"),
                fetched_at=p.get("fetched_at", manifest["taken_on"]),
            )
            for p in manifest["pages"]
        ]
        snap = Snapshot(manifest["snapshot_id"], manifest["taken_on"], pages)
        print(f"  ingested {snap.snapshot_id} ({len(snap.pages)} pages)")
        candidate.ingest(snap)
    ks = candidate
    print("Knowledge system ready.")
    return ks


@app.before_request
def _ensure_system_ready():
    if ks is None:
        try:
            with _system_lock:
                if ks is None:
                    load_system()
        except Exception:
            logger.exception("Knowledge system initialization failed")
            return jsonify({"error": "Knowledge system is unavailable"}), 503


@app.errorhandler(413)
def _payload_too_large(_error):
    return jsonify({"error": "Request body is too large"}), 413


@app.after_request
def _add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'",
    )
    return response


@app.route("/")
def index():
    return Response(HTML_PAGE, content_type="text/html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "A non-empty question is required"}), 400
    question = question.strip()
    if len(question) > 2000:
        return jsonify({"error": "Question is too long"}), 413

    as_of = data.get("as_of") or None
    if as_of is not None:
        if not isinstance(as_of, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
            return jsonify({"error": "as_of must use YYYY-MM-DD"}), 400
        try:
            date.fromisoformat(as_of)
        except ValueError:
            return jsonify({"error": "as_of must be a valid calendar date"}), 400

    t = time.perf_counter()
    try:
        ans = ks.answer(question, as_of=as_of)
    except Exception as exc:
        logger.error("Question processing failed (%s)", type(exc).__name__)
        return jsonify({"error": "Could not answer the question right now"}), 503
    elapsed = time.perf_counter() - t

    return jsonify({
        "text": ans.text,
        "confident": ans.confident,
        "sources": [{"url": s.url, "snapshot_id": s.snapshot_id} for s in ans.sources],
        "conflict_note": ans.conflict_note,
        "elapsed": round(elapsed, 2),
    })


@app.route("/api/conflicts")
def conflicts():
    cs = ks.conflicts()
    return jsonify([{
        "topic": c.topic,
        "values": c.values,
        "reason": c.reason,
        "resolution": c.resolution,
    } for c in cs[:20]])


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dupaco Knowledge Assistant</title>
<style>
  :root {
    --bg: #f7f8fa;
    --surface: #ffffff;
    --border: #e2e5ea;
    --fg: #1a1d23;
    --fg-dim: #6b7280;
    --accent: #1a6b4a;
    --accent-light: #e8f5ee;
    --user-bg: #1a6b4a;
    --user-fg: #ffffff;
    --radius: 12px;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111318;
      --surface: #1a1d24;
      --border: #2a2d35;
      --fg: #e4e6ea;
      --fg-dim: #8b8f98;
      --accent: #3dba7a;
      --accent-light: #1a2e24;
      --user-bg: #2d8a5e;
      --user-fg: #ffffff;
    }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--fg);
    display: flex;
    flex-direction: column;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  header .logo {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 16px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header .subtitle { font-size: 13px; color: var(--fg-dim); }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .chat-area::-webkit-scrollbar { width: 6px; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .welcome {
    text-align: center;
    padding: 40px 20px;
    color: var(--fg-dim);
  }
  .welcome h2 { font-size: 20px; color: var(--fg); margin-bottom: 8px; }
  .welcome p { font-size: 14px; line-height: 1.6; max-width: 420px; margin: 0 auto; }
  .suggestions {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-top: 16px;
  }
  .suggestions button {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 8px 14px;
    font-size: 13px;
    color: var(--fg);
    cursor: pointer;
    font-family: var(--font);
    transition: border-color 0.15s;
  }
  .suggestions button:hover { border-color: var(--accent); color: var(--accent); }

  .msg { display: flex; gap: 10px; max-width: 720px; margin: 0 auto; width: 100%; }
  .msg.user { justify-content: flex-end; }
  .msg.user .bubble {
    background: var(--user-bg); color: var(--user-fg);
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    max-width: 75%;
  }
  .msg.bot .bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
    max-width: 85%;
  }
  .bubble {
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble .meta {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--fg-dim);
  }
  .bubble .meta a {
    color: var(--accent);
    text-decoration: none;
    word-break: break-all;
  }
  .bubble .meta a:hover { text-decoration: underline; }
  .bubble .confidence {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 500;
  }
  .confidence.high { background: var(--accent-light); color: var(--accent); }
  .confidence.low { background: #fef3c7; color: #92400e; }
  .bubble .conflict {
    margin-top: 6px;
    font-size: 12px;
    color: #b45309;
    font-style: italic;
  }

  .typing { display: flex; gap: 4px; padding: 12px 16px; }
  .typing span {
    width: 8px; height: 8px;
    background: var(--fg-dim);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  .input-area {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 16px;
    flex-shrink: 0;
  }
  .input-row {
    max-width: 720px;
    margin: 0 auto;
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  .input-row input[type="text"] {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 14px;
    font-family: var(--font);
    background: var(--bg);
    color: var(--fg);
    outline: none;
    transition: border-color 0.15s;
  }
  .input-row input[type="text"]:focus { border-color: var(--accent); }
  .input-row input[type="date"] {
    padding: 12px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
    font-family: var(--font);
    background: var(--bg);
    color: var(--fg);
    outline: none;
    width: 140px;
  }
  .input-row button {
    padding: 12px 20px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    font-family: var(--font);
    transition: opacity 0.15s;
  }
  .input-row button:hover { opacity: 0.9; }
  .input-row button:disabled { opacity: 0.5; cursor: not-allowed; }

  .date-toggle {
    font-size: 12px;
    color: var(--fg-dim);
    text-align: center;
    margin-top: 6px;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
  }
  .date-toggle a {
    color: var(--accent);
    cursor: pointer;
    text-decoration: none;
  }
</style>
</head>
<body>

<header>
  <div class="logo">D</div>
  <div>
    <h1>Dupaco Knowledge Assistant</h1>
    <div class="subtitle">Ask questions about Dupaco Credit Union</div>
  </div>
</header>

<div class="chat-area" id="chat">
  <div class="welcome" id="welcome">
    <h2>How can I help you today?</h2>
    <p>Ask me anything about Dupaco's products, rates, branches, and services. I'll answer from the latest website snapshots.</p>
    <div class="suggestions">
      <button onclick="askSuggestion(this)">What savings accounts do you offer?</button>
      <button onclick="askSuggestion(this)">What is the auto loan rate?</button>
      <button onclick="askSuggestion(this)">Where is the Galena branch?</button>
      <button onclick="askSuggestion(this)">What credit cards are available?</button>
    </div>
  </div>
</div>

<div class="input-area">
  <form class="input-row" id="form" onsubmit="handleSubmit(event)">
    <input type="text" id="question" placeholder="Ask a question..." autocomplete="off">
    <input type="date" id="asof" style="display:none" title="Answer as of this date">
    <button type="submit" id="sendbtn">Send</button>
  </form>
  <div class="date-toggle">
    <a onclick="toggleDate()" id="datelink">+ Add as-of date for temporal queries</a>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const qInput = document.getElementById('question');
const asofInput = document.getElementById('asof');
const sendBtn = document.getElementById('sendbtn');
const welcome = document.getElementById('welcome');
let busy = false;

function askSuggestion(btn) {
  qInput.value = btn.textContent;
  handleSubmit(new Event('submit'));
}

function toggleDate() {
  const shown = asofInput.style.display !== 'none';
  asofInput.style.display = shown ? 'none' : '';
  document.getElementById('datelink').textContent = shown
    ? '+ Add as-of date for temporal queries'
    : '- Remove as-of date';
}

function addMsg(role, html) {
  if (welcome) welcome.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="bubble">' + html + '</div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing';
  div.innerHTML = '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function safeSourceUrl(raw) {
  try {
    const url = new URL(raw);
    if (url.protocol !== 'https:' || !['www.dupaco.com', 'dupaco.com'].includes(url.hostname.toLowerCase())) return null;
    return url.href;
  } catch (_) {
    return null;
  }
}

async function handleSubmit(e) {
  e.preventDefault();
  const q = qInput.value.trim();
  if (!q || busy) return;

  const asOf = asofInput.value || null;
  addMsg('user', escHtml(q) + (asOf ? '<br><small style="opacity:0.7">as of ' + escHtml(asOf) + '</small>' : ''));
  qInput.value = '';
  busy = true;
  sendBtn.disabled = true;
  addTyping();

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q, as_of: asOf})
    });
    const data = await res.json();
    removeTyping();

    let html = escHtml(data.text);
    let meta = '';
    if (data.sources && data.sources.length) {
      meta += '<div style="margin-top:4px"><strong>Sources:</strong></div>';
      data.sources.forEach(s => {
        const href = safeSourceUrl(s.url);
        if (!href) return;
        const short = href.replace('https://www.dupaco.com','');
        meta += '<a href="' + escHtml(href) + '" target="_blank" rel="noopener noreferrer">' + escHtml(short) + '</a><br>';
      });
    }
    const confClass = data.confident ? 'high' : 'low';
    const confLabel = data.confident ? 'Confident' : 'Low confidence';
    meta += '<span class="confidence ' + confClass + '">' + confLabel + '</span>';
    meta += ' <span style="opacity:0.6">' + data.elapsed + 's</span>';

    if (data.conflict_note) {
      meta += '<div class="conflict">' + escHtml(data.conflict_note) + '</div>';
    }

    html += '<div class="meta">' + meta + '</div>';
    addMsg('bot', html);
  } catch (err) {
    removeTyping();
    addMsg('bot', 'Something went wrong. Is the server running?');
  }
  busy = false;
  sendBtn.disabled = false;
  qInput.focus();
}

qInput.focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Loading knowledge system...")
    load_system()
    print("\nStarting server at http://localhost:5000")
    # Flask's built-in server is for local development only. Deploy behind a
    # production WSGI server and an authenticated, rate-limited gateway.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        use_reloader=False,
    )
