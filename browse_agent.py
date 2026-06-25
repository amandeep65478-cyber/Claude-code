#!/usr/bin/env python3
"""
Browse Agent — an AI agent powered by Claude that can browse the web.
Uses Playwright for browser automation + Anthropic API for reasoning.

Usage:
    python browse_agent.py
    python browse_agent.py --task "Find the latest news about AI"
"""

import os
import sys
import json
import argparse
import textwrap
from typing import Optional
import anthropic
from playwright.sync_api import sync_playwright, Page, Browser

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
CHROMIUM_PATH = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
VIEWPORT = {"width": 1280, "height": 800}
MAX_CONTENT_LENGTH = 8000   # chars sent back to Claude per page
MAX_ITERATIONS = 20         # guard against infinite loops


# ── Browser helpers ─────────────────────────────────────────────────────────────

def get_page_text(page: Page) -> str:
    """Extract visible text from the current page."""
    try:
        text = page.evaluate("""() => {
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                {
                    acceptNode: n => {
                        const p = n.parentElement;
                        if (!p) return NodeFilter.FILTER_REJECT;
                        const tag = p.tagName.toLowerCase();
                        if (['script','style','noscript','svg'].includes(tag))
                            return NodeFilter.FILTER_REJECT;
                        const style = getComputedStyle(p);
                        if (style.display === 'none' || style.visibility === 'hidden')
                            return NodeFilter.FILTER_REJECT;
                        return NodeFilter.FILTER_ACCEPT;
                    }
                }
            );
            const parts = [];
            let node;
            while ((node = walker.nextNode())) {
                const t = node.textContent.trim();
                if (t) parts.push(t);
            }
            return parts.join(' ');
        }""")
        return text[:MAX_CONTENT_LENGTH]
    except Exception as e:
        return f"[Error extracting text: {e}]"


def get_links(page: Page) -> list[dict]:
    """Get all links visible on the page."""
    try:
        links = page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a[href]'))
                .filter(a => a.textContent.trim())
                .slice(0, 30)
                .map(a => ({
                    text: a.textContent.trim().slice(0, 80),
                    href: a.href
                }));
        }""")
        return links
    except Exception:
        return []


# ── Tool definitions (sent to Claude) ──────────────────────────────────────────

TOOLS = [
    {
        "name": "navigate",
        "description": "Go to a URL in the browser. Returns the page title and visible text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to navigate to (must include https://)"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "search_google",
        "description": "Search Google for a query and return the search results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_page_content",
        "description": "Get the current page's full visible text content and list of links.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "click_link",
        "description": "Click a link on the current page by matching its text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "link_text": {"type": "string", "description": "Partial or full text of the link to click"}
            },
            "required": ["link_text"]
        }
    },
    {
        "name": "scroll_down",
        "description": "Scroll down the current page to load more content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "Pixels to scroll (default 800)", "default": 800}
            }
        }
    },
    {
        "name": "go_back",
        "description": "Go back to the previous page in browser history.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "type_and_search",
        "description": "Find an input/search box on the page, type text into it, and submit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type and search"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "get_current_url",
        "description": "Get the URL and title of the currently loaded page.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "done",
        "description": "Signal that you have finished the task and provide your final answer/summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your complete final answer or summary of findings"}
            },
            "required": ["answer"]
        }
    }
]


# ── Tool execution ──────────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict, page: Page) -> str:
    try:
        if name == "navigate":
            url = inputs["url"]
            if not url.startswith("http"):
                url = "https://" + url
            print(f"  → Navigating to: {url}")
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            title = page.title()
            text = get_page_text(page)
            return f"[Page: {title}]\n[URL: {page.url}]\n\n{text}"

        elif name == "search_google":
            query = inputs["query"]
            print(f"  → Searching Google: {query}")
            encoded = query.replace(" ", "+")
            page.goto(f"https://www.google.com/search?q={encoded}", timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            text = get_page_text(page)
            links = get_links(page)
            link_str = "\n".join(f"  [{i+1}] {l['text']} → {l['href']}" for i, l in enumerate(links[:15]))
            return f"[Google Search Results for: {query}]\n\n{text[:4000]}\n\nLINKS:\n{link_str}"

        elif name == "get_page_content":
            print(f"  → Reading page content")
            title = page.title()
            text = get_page_text(page)
            links = get_links(page)
            link_str = "\n".join(f"  [{i+1}] {l['text']} → {l['href']}" for i, l in enumerate(links))
            return f"[Page: {title}]\n[URL: {page.url}]\n\nCONTENT:\n{text}\n\nLINKS ON PAGE:\n{link_str}"

        elif name == "click_link":
            link_text = inputs["link_text"]
            print(f"  → Clicking link: {link_text}")
            try:
                page.click(f"a:has-text('{link_text}')", timeout=5000)
            except Exception:
                # Try partial match via JS
                page.evaluate(f"""() => {{
                    const links = Array.from(document.querySelectorAll('a'));
                    const match = links.find(a => a.textContent.toLowerCase().includes('{link_text.lower()}'));
                    if (match) match.click();
                }}""")
            page.wait_for_timeout(1500)
            title = page.title()
            text = get_page_text(page)
            return f"[Clicked link, now on: {title}]\n[URL: {page.url}]\n\n{text}"

        elif name == "scroll_down":
            amount = inputs.get("amount", 800)
            print(f"  → Scrolling down {amount}px")
            page.evaluate(f"window.scrollBy(0, {amount})")
            page.wait_for_timeout(500)
            text = get_page_text(page)
            return f"[Scrolled down {amount}px]\n\n{text}"

        elif name == "go_back":
            print(f"  → Going back")
            page.go_back(timeout=10000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            title = page.title()
            return f"[Went back to: {title}]\n[URL: {page.url}]"

        elif name == "type_and_search":
            text = inputs["text"]
            print(f"  → Typing and searching: {text}")
            selectors = ['input[type="search"]', 'input[type="text"]', 'input[name="q"]', 'textarea', 'input:not([type="hidden"])']
            found = False
            for sel in selectors:
                try:
                    page.fill(sel, text, timeout=3000)
                    page.keyboard.press("Enter")
                    found = True
                    break
                except Exception:
                    continue
            if not found:
                return "[Could not find a search/input box on this page]"
            page.wait_for_timeout(1500)
            title = page.title()
            content = get_page_text(page)
            return f"[Searched for '{text}', now on: {title}]\n\n{content}"

        elif name == "get_current_url":
            return f"URL: {page.url}\nTitle: {page.title()}"

        elif name == "done":
            return "__DONE__"

        else:
            return f"[Unknown tool: {name}]"

    except Exception as e:
        return f"[Tool error ({name}): {e}]"


# ── Agent loop ──────────────────────────────────────────────────────────────────

def run_agent(task: str):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_prompt = textwrap.dedent("""
        You are a powerful web browsing agent. You have full control of a real Chromium browser.

        Your job is to complete the user's task by browsing the web intelligently.

        Guidelines:
        - Use search_google to find information you don't know the URL for
        - Use navigate to go directly to known URLs
        - Use get_page_content to read what's currently on screen
        - Use click_link to follow links to get more detail
        - Use scroll_down if you think there's more content below
        - Use go_back if you ended up on the wrong page
        - Use type_and_search to interact with search boxes on pages
        - Always use the done tool when you have found the answer — include a complete, detailed summary
        - Be thorough: visit multiple sources if needed
        - Don't get stuck — if one approach fails, try another
    """).strip()

    messages = [{"role": "user", "content": task}]

    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}\n")

    with sync_playwright() as pw:
        browser: Browser = pw.chromium.launch(
            executable_path=CHROMIUM_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page: Page = browser.new_page(viewport=VIEWPORT)
        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"})

        final_answer = None

        for iteration in range(MAX_ITERATIONS):
            print(f"\n[Iteration {iteration + 1}/{MAX_ITERATIONS}]")

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=TOOLS,
                messages=messages
            )

            # Collect assistant content
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract any final text
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nClaude: {block.text}")
                break

            # Process tool calls
            tool_results = []
            all_done = False

            for block in assistant_content:
                if block.type == "text" and block.text:
                    print(f"\nClaude: {block.text}")

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"\n[Tool: {tool_name}] {json.dumps(tool_input)[:120]}")

                    result = execute_tool(tool_name, tool_input, page)

                    if result == "__DONE__":
                        final_answer = tool_input.get("answer", "")
                        all_done = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Task marked as complete."
                        })
                    else:
                        # Truncate long results
                        if len(result) > MAX_CONTENT_LENGTH:
                            result = result[:MAX_CONTENT_LENGTH] + "\n...[truncated]"
                        print(f"  Result preview: {result[:200]}...")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if all_done:
                break

        browser.close()

    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print('='*60)
    if final_answer:
        print(final_answer)
    else:
        print("[Agent finished without calling done()]")
    print('='*60)

    return final_answer


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Browse Agent powered by Claude")
    parser.add_argument("--task", "-t", type=str, help="Task to perform (optional, will prompt if not given)")
    parser.add_argument("--api-key", type=str, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    if args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: No API key found.")
        print("Set ANTHROPIC_API_KEY environment variable or pass --api-key")
        sys.exit(1)

    if args.task:
        task = args.task
    else:
        print("Browse Agent — powered by Claude + Playwright")
        print("Type your task and press Enter. Type 'quit' to exit.\n")
        while True:
            task = input("Task> ").strip()
            if task.lower() in ("quit", "exit", "q"):
                break
            if task:
                run_agent(task)

if __name__ == "__main__":
    main()
