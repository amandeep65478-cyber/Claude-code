#!/usr/bin/env python3
"""
Browse Agent Web UI — FastAPI + SSE + Playwright + Claude
Open http://localhost:8000 in your browser and give the agent a task.
"""

import os
import json
import base64
import asyncio
import threading
import queue
import textwrap
from typing import Optional
from pathlib import Path

import anthropic
from playwright.sync_api import sync_playwright, Page, Browser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
CHROMIUM_PATH = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
VIEWPORT = {"width": 1280, "height": 800}
MAX_CONTENT_LENGTH = 6000
MAX_ITERATIONS = 20

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── HTML UI ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Browse Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; min-height: 100vh; }

  header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 20px 30px;
    border-bottom: 1px solid #2a2a4a;
    display: flex; align-items: center; gap: 12px;
  }
  header h1 { font-size: 1.5rem; color: #a78bfa; font-weight: 700; }
  header span { font-size: 0.85rem; color: #666; }

  .container { display: grid; grid-template-columns: 1fr 1fr; gap: 0; height: calc(100vh - 73px); }

  /* Left panel */
  .left { display: flex; flex-direction: column; border-right: 1px solid #2a2a4a; }

  .task-bar {
    padding: 16px;
    background: #13131a;
    border-bottom: 1px solid #2a2a4a;
    display: flex; gap: 10px;
  }
  .task-bar input {
    flex: 1; padding: 10px 14px;
    background: #1e1e2e; border: 1px solid #3a3a5a;
    border-radius: 8px; color: #e0e0e0; font-size: 0.95rem;
    outline: none;
  }
  .task-bar input:focus { border-color: #a78bfa; }
  .task-bar button {
    padding: 10px 20px; background: #7c3aed;
    color: white; border: none; border-radius: 8px;
    font-size: 0.95rem; cursor: pointer; font-weight: 600;
    transition: background 0.2s;
  }
  .task-bar button:hover { background: #6d28d9; }
  .task-bar button:disabled { background: #3a3a5a; cursor: not-allowed; }

  .log-panel {
    flex: 1; overflow-y: auto; padding: 16px;
    font-family: 'Courier New', monospace; font-size: 0.82rem;
    line-height: 1.6;
  }

  .log-entry { margin-bottom: 8px; padding: 8px 12px; border-radius: 6px; animation: fadeIn 0.2s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  .log-tool { background: #1a1a2e; border-left: 3px solid #a78bfa; color: #c4b5fd; }
  .log-result { background: #0f1a0f; border-left: 3px solid #34d399; color: #6ee7b7; }
  .log-claude { background: #1a1a1a; border-left: 3px solid #60a5fa; color: #93c5fd; }
  .log-done { background: #1a1205; border-left: 3px solid #fbbf24; color: #fde68a; }
  .log-error { background: #1a0f0f; border-left: 3px solid #f87171; color: #fca5a5; }
  .log-info { background: #13131a; border-left: 3px solid #4b5563; color: #9ca3af; }

  .log-label { font-weight: 700; margin-bottom: 2px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; }

  /* Right panel — browser preview */
  .right { display: flex; flex-direction: column; background: #0a0a0f; }
  .browser-bar {
    padding: 10px 16px; background: #13131a;
    border-bottom: 1px solid #2a2a4a;
    display: flex; align-items: center; gap: 10px;
  }
  .browser-dots span { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }
  .dot-r { background: #f87171; } .dot-y { background: #fbbf24; } .dot-g { background: #34d399; }
  #url-display { flex: 1; padding: 5px 10px; background: #1e1e2e; border: 1px solid #3a3a5a; border-radius: 6px; font-size: 0.8rem; color: #9ca3af; font-family: monospace; }
  .browser-content { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  #screenshot { max-width: 100%; max-height: 100%; object-fit: contain; display: none; border: 1px solid #2a2a4a; }
  .placeholder { text-align: center; color: #3a3a5a; }
  .placeholder svg { margin-bottom: 12px; }
  .placeholder p { font-size: 0.9rem; }

  .status-bar {
    padding: 8px 16px; background: #13131a;
    border-top: 1px solid #2a2a4a;
    font-size: 0.78rem; color: #6b7280;
    display: flex; justify-content: space-between;
  }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4b5563; display: inline-block; margin-right: 6px; }
  #status-dot.active { background: #34d399; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .final-answer {
    margin: 12px 0; padding: 14px;
    background: #1a1205; border: 1px solid #92400e;
    border-radius: 8px; color: #fde68a;
    font-family: 'Segoe UI', sans-serif; font-size: 0.9rem;
    line-height: 1.7; white-space: pre-wrap;
  }
</style>
</head>
<body>
<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
  <h1>Browse Agent</h1>
  <span>Powered by Claude + Playwright</span>
</header>

<div class="container">
  <!-- Left: task + log -->
  <div class="left">
    <div class="task-bar">
      <input id="task-input" type="text" placeholder="e.g. Find top AI news today, summarize Python 3.13 features..." />
      <button id="run-btn" onclick="runAgent()">Run</button>
    </div>
    <div class="log-panel" id="log"></div>
  </div>

  <!-- Right: browser preview -->
  <div class="right">
    <div class="browser-bar">
      <div class="browser-dots">
        <span class="dot-r"></span><span class="dot-y"></span><span class="dot-g"></span>
      </div>
      <div id="url-display">about:blank</div>
    </div>
    <div class="browser-content">
      <img id="screenshot" alt="Browser Screenshot" />
      <div class="placeholder" id="placeholder">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#3a3a5a" stroke-width="1.5">
          <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
        </svg>
        <p>Browser preview will appear here</p>
      </div>
    </div>
    <div class="status-bar">
      <span><span id="status-dot"></span><span id="status-text">Ready</span></span>
      <span id="iter-count"></span>
    </div>
  </div>
</div>

<script>
let running = false;

function addLog(type, label, content) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  div.className = `log-entry log-${type}`;
  div.innerHTML = `<div class="log-label">${label}</div>${escHtml(content)}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setStatus(text, active=false) {
  document.getElementById('status-text').textContent = text;
  document.getElementById('status-dot').className = active ? 'active' : '';
}

async function runAgent() {
  if (running) return;
  const task = document.getElementById('task-input').value.trim();
  if (!task) return;

  running = true;
  document.getElementById('run-btn').disabled = true;
  document.getElementById('log').innerHTML = '';
  document.getElementById('screenshot').style.display = 'none';
  document.getElementById('placeholder').style.display = 'flex';
  document.getElementById('url-display').textContent = 'Starting...';
  setStatus('Running...', true);

  addLog('info', 'Task', task);

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\\n').filter(l => l.startsWith('data: '));

      for (const line of lines) {
        try {
          const ev = JSON.parse(line.slice(6));

          if (ev.type === 'tool') {
            addLog('tool', `Tool: ${ev.name}`, ev.input);
            setStatus(`Using: ${ev.name}`, true);
          } else if (ev.type === 'result') {
            addLog('result', 'Result', ev.content.slice(0, 300) + (ev.content.length > 300 ? '...' : ''));
          } else if (ev.type === 'claude') {
            addLog('claude', 'Claude', ev.content);
          } else if (ev.type === 'screenshot') {
            const img = document.getElementById('screenshot');
            img.src = 'data:image/png;base64,' + ev.data;
            img.style.display = 'block';
            document.getElementById('placeholder').style.display = 'none';
          } else if (ev.type === 'url') {
            document.getElementById('url-display').textContent = ev.url;
          } else if (ev.type === 'iteration') {
            document.getElementById('iter-count').textContent = `Step ${ev.n}/${ev.max}`;
          } else if (ev.type === 'done') {
            const div = document.createElement('div');
            div.className = 'final-answer';
            div.innerHTML = '<strong>Final Answer:</strong>\\n' + escHtml(ev.answer);
            document.getElementById('log').appendChild(div);
            document.getElementById('log').scrollTop = 99999;
            setStatus('Done', false);
          } else if (ev.type === 'error') {
            addLog('error', 'Error', ev.message);
            setStatus('Error', false);
          }
        } catch (e) {}
      }
    }
  } catch (e) {
    addLog('error', 'Error', e.message);
    setStatus('Error', false);
  }

  running = false;
  document.getElementById('run-btn').disabled = false;
  if (document.getElementById('status-text').textContent === 'Running...') {
    setStatus('Finished', false);
  }
}

document.getElementById('task-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAgent();
});
</script>
</body>
</html>
"""


# ── Tool helpers ────────────────────────────────────────────────────────────────

def get_page_text(page: Page) -> str:
    try:
        text = page.evaluate("""() => {
            const walker = document.createTreeWalker(
                document.body, NodeFilter.SHOW_TEXT,
                { acceptNode: n => {
                    const p = n.parentElement;
                    if (!p) return NodeFilter.FILTER_REJECT;
                    const tag = p.tagName.toLowerCase();
                    if (['script','style','noscript','svg'].includes(tag)) return NodeFilter.FILTER_REJECT;
                    const s = getComputedStyle(p);
                    if (s.display==='none'||s.visibility==='hidden') return NodeFilter.FILTER_REJECT;
                    return NodeFilter.FILTER_ACCEPT;
                }}
            );
            const parts=[]; let node;
            while((node=walker.nextNode())){const t=node.textContent.trim();if(t)parts.push(t);}
            return parts.join(' ');
        }""")
        return text[:MAX_CONTENT_LENGTH]
    except Exception as e:
        return f"[Error: {e}]"


def get_links(page: Page) -> list:
    try:
        return page.evaluate("""() =>
            Array.from(document.querySelectorAll('a[href]'))
                .filter(a=>a.textContent.trim())
                .slice(0,25)
                .map(a=>({text:a.textContent.trim().slice(0,80), href:a.href}))
        """)
    except Exception:
        return []


def screenshot_b64(page: Page) -> Optional[str]:
    try:
        data = page.screenshot(type="png", full_page=False)
        return base64.b64encode(data).decode()
    except Exception:
        return None


TOOLS = [
    {"name": "navigate", "description": "Go to a URL. Returns page title and text content.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "search_google", "description": "Search Google and return results.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_page_content", "description": "Get current page text and links.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "click_link", "description": "Click a link by its text.",
     "input_schema": {"type": "object", "properties": {"link_text": {"type": "string"}}, "required": ["link_text"]}},
    {"name": "scroll_down", "description": "Scroll down the page.",
     "input_schema": {"type": "object", "properties": {"amount": {"type": "integer", "default": 800}}}},
    {"name": "go_back", "description": "Go back in browser history.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "type_and_search", "description": "Type in a search box and submit.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "done", "description": "Finish and return the final answer.",
     "input_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
]

SYSTEM = textwrap.dedent("""
    You are a powerful web browsing agent with full control of a real Chromium browser.
    Complete the user's task by browsing the web intelligently.
    - Use search_google to find things you don't know the URL for
    - Use navigate for direct URLs
    - Visit multiple sources when needed for accuracy
    - Use done when you have a complete answer — include all details
""").strip()


def execute_tool(name, inputs, page, emit):
    try:
        if name == "navigate":
            url = inputs["url"]
            if not url.startswith("http"): url = "https://" + url
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            text = get_page_text(page)
            return f"[Page: {page.title()}]\n[URL: {page.url}]\n\n{text}"

        elif name == "search_google":
            q = inputs["query"]
            page.goto(f"https://www.google.com/search?q={q.replace(' ', '+')}", timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            text = get_page_text(page)
            links = get_links(page)
            link_str = "\n".join(f"  [{i+1}] {l['text']} → {l['href']}" for i, l in enumerate(links[:15]))
            return f"[Google: {q}]\n\n{text[:4000]}\n\nLINKS:\n{link_str}"

        elif name == "get_page_content":
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            text = get_page_text(page)
            links = get_links(page)
            link_str = "\n".join(f"  [{i+1}] {l['text']} → {l['href']}" for i, l in enumerate(links))
            return f"[Page: {page.title()}]\n[URL: {page.url}]\n\nCONTENT:\n{text}\n\nLINKS:\n{link_str}"

        elif name == "click_link":
            lt = inputs["link_text"]
            try:
                page.click(f"a:has-text('{lt}')", timeout=5000)
            except Exception:
                page.evaluate(f"()=>{{const a=Array.from(document.querySelectorAll('a')).find(a=>a.textContent.toLowerCase().includes('{lt.lower()}'));if(a)a.click();}}")
            page.wait_for_timeout(1200)
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            text = get_page_text(page)
            return f"[Clicked '{lt}', now: {page.title()}]\n[URL: {page.url}]\n\n{text}"

        elif name == "scroll_down":
            amt = inputs.get("amount", 800)
            page.evaluate(f"window.scrollBy(0,{amt})")
            page.wait_for_timeout(400)
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            return f"[Scrolled {amt}px]\n\n{get_page_text(page)}"

        elif name == "go_back":
            page.go_back(timeout=10000, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            return f"[Back to: {page.title()}]\n[URL: {page.url}]"

        elif name == "type_and_search":
            text = inputs["text"]
            for sel in ['input[type="search"]','input[name="q"]','input[type="text"]','textarea']:
                try:
                    page.fill(sel, text, timeout=3000)
                    page.keyboard.press("Enter")
                    break
                except Exception:
                    continue
            page.wait_for_timeout(1200)
            emit({"type": "url", "url": page.url})
            sc = screenshot_b64(page)
            if sc: emit({"type": "screenshot", "data": sc})
            return f"[Searched '{text}']\n\n{get_page_text(page)}"

        elif name == "done":
            return "__DONE__"

        return f"[Unknown tool: {name}]"
    except Exception as e:
        return f"[Tool error ({name}): {e}]"


# ── Agent in a thread (SSE stream) ──────────────────────────────────────────────

def agent_thread(task: str, q: queue.Queue):
    def emit(ev):
        q.put(ev)

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            emit({"type": "error", "message": "ANTHROPIC_API_KEY not set"})
            q.put(None)
            return

        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": task}]

        with sync_playwright() as pw:
            browser: Browser = pw.chromium.launch(
                executable_path=CHROMIUM_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            page: Page = browser.new_page(viewport=VIEWPORT)
            page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"})

            for iteration in range(MAX_ITERATIONS):
                emit({"type": "iteration", "n": iteration + 1, "max": MAX_ITERATIONS})

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    tools=TOOLS,
                    messages=messages
                )

                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                if response.stop_reason == "end_turn":
                    for block in assistant_content:
                        if hasattr(block, "text") and block.text:
                            emit({"type": "claude", "content": block.text})
                    break

                tool_results = []
                all_done = False

                for block in assistant_content:
                    if block.type == "text" and block.text:
                        emit({"type": "claude", "content": block.text})
                    elif block.type == "tool_use":
                        emit({"type": "tool", "name": block.name, "input": json.dumps(block.input)[:200]})
                        result = execute_tool(block.name, block.input, page, emit)

                        if result == "__DONE__":
                            all_done = True
                            final = block.input.get("answer", "")
                            emit({"type": "done", "answer": final})
                            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Done."})
                        else:
                            if len(result) > MAX_CONTENT_LENGTH:
                                result = result[:MAX_CONTENT_LENGTH] + "\n...[truncated]"
                            emit({"type": "result", "content": result})
                            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                if all_done:
                    break

            browser.close()

    except Exception as e:
        emit({"type": "error", "message": str(e)})

    q.put(None)  # signal end


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    task = body.get("task", "").strip()
    if not task:
        return {"error": "No task provided"}

    q: queue.Queue = queue.Queue()

    t = threading.Thread(target=agent_thread, args=(task, q), daemon=True)
    t.start()

    async def stream():
        while True:
            try:
                ev = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 60)
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type':'heartbeat'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n Browse Agent running at http://localhost:{port}")
    print(f" Set ANTHROPIC_API_KEY before starting.\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
