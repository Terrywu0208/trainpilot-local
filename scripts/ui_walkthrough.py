#!/usr/bin/env python3
"""ui_walkthrough.py - drive the real TrainPilot-Local web UI in a browser.

End-to-end UI verification with Playwright + Chromium (headless):

  1. load the console, check the local/CPU-only framing
  2. submit the canonical request, approve Gate 1 in the modal
  3. wait for the VERIFIED SUCCESS verdict and the accuracy badge
  4. submit the same-session follow-up, approve, verify session context
  5. open Failure Memory, Benchmark Dashboard and Replay views

A screenshot is saved at every stage. Exit code 0 means every assertion
passed. Prerequisites: the web app is running (see docs/OPERATIONS.md
section 6) and Playwright + Chromium are installed:

    pip install playwright && python -m playwright install chromium
    python scripts/ui_walkthrough.py --base http://127.0.0.1:8126 \
        --shots docs/images

No network access and no paid service is used at any point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CANONICAL = "Fine-tune DistilBERT on SST-2 with 2 CPU workers"
FOLLOWUP = "Same setup as before, but run 3 epochs this time"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8126")
    ap.add_argument("--shots", default="/tmp/ui-shots")
    args = ap.parse_args()
    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    results: list[bool] = []

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:
            # Fall back to a cached full-Chromium build when only the
            # headless shell is missing from the Playwright cache.
            import glob
            candidates = sorted(glob.glob(
                str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux/chrome")))
            if not candidates:
                raise
            browser = pw.chromium.launch(executable_path=candidates[-1])
        page = browser.new_page(viewport={"width": 1440, "height": 950})

        # 1. Console loads with the local CPU-only framing.
        page.goto(args.base, wait_until="networkidle")
        results.append(check("page title", "TrainPilot" in page.title(), page.title()))
        badge = page.locator("text=Local · CPU-only demo").first
        results.append(check("CPU-only badge visible", badge.is_visible()))
        results.append(check("stub LLM chip visible",
                             page.locator("text=stub").first.is_visible()))
        page.screenshot(path=str(shots / "01-console-initial.png"))

        # 2. Canonical request -> Gate 1 approval card in the chat pane.
        page.click("#preset-canonical")
        results.append(check("preset fills the request box",
                             page.input_value("#request-text") == CANONICAL))
        page.click("#btn-run")
        page.wait_for_function(
            "document.querySelector('#chat-messages').textContent.includes('Fine-tune DistilBERT')",
            timeout=15000)
        results.append(check("chat shows the user bubble",
                             CANONICAL in page.inner_text("#chat-messages")))
        page.wait_for_selector("#gate-modal", state="visible", timeout=30000)
        results.append(check("gate card is docked above the composer",
                             page.locator("#gate-dock #gate-modal").is_visible()))
        gate_title = page.inner_text("#gate-title")
        results.append(check("Gate 1 modal appears", "launch" in gate_title.lower(),
                             gate_title.strip()))
        page.screenshot(path=str(shots / "02-gate1-approval.png"))

        # 3. Approve -> the real run finishes VERIFIED SUCCESS at 0.8771.
        page.click("#gate-approve")
        page.wait_for_selector("#verdict:not(:empty)", timeout=180000)
        page.wait_for_function(
            "document.querySelector('#verdict').textContent.includes('VERIFIED')",
            timeout=180000)
        verdict = page.inner_text("#verdict")
        results.append(check("verdict is VERIFIED SUCCESS",
                             "VERIFIED SUCCESS" in verdict, verdict.strip()[:90]))
        acc = page.inner_text("#vb-accuracy")
        results.append(check("accuracy badge shows 0.8771", "0.8771" in acc, acc.strip()))
        results.append(check("chat contains the verdict card",
                             "VERIFIED SUCCESS" in page.inner_text("#chat-messages")))
        done_nodes = page.locator("#plan-nodes .node.done, #plan-nodes .done").count()
        results.append(check("pipeline nodes completed", done_nodes >= 10,
                             f"{done_nodes} nodes done"))
        page.screenshot(path=str(shots / "03-verified-success.png"), full_page=True)

        # 4. Same-session follow-up reuses the previous run's context.
        page.click("#preset-followup")
        results.append(check("follow-up preset fills the box",
                             page.input_value("#request-text") == FOLLOWUP))
        page.click("#btn-run")
        page.wait_for_selector("#gate-modal", state="visible", timeout=30000)
        page.click("#gate-approve")
        page.wait_for_function(
            "document.querySelector('#verdict').textContent.includes('VERIFIED')",
            timeout=180000)
        ctx = page.inner_text("#session-ctx-text")
        results.append(check("session context indicator active",
                             "active" in ctx.lower() or "used" in ctx.lower(),
                             ctx.strip()))
        page.screenshot(path=str(shots / "04-followup-run.png"), full_page=True)

        # 5a. Failure Memory view renders its table.
        page.click('[data-view="failures"]')
        page.wait_for_selector("#failure-rows", timeout=10000)
        page.wait_for_timeout(800)
        count_text = page.inner_text("#failure-count")
        results.append(check("failure memory renders", "of" in count_text, count_text))
        page.screenshot(path=str(shots / "05-failure-memory.png"))

        # 5b. Benchmark Dashboard renders charts from reports/final.
        page.click('[data-view="benchmark"]')
        page.wait_for_timeout(1200)
        body = page.inner_text("#benchmark-body")
        results.append(check("dashboard shows both lab systems",
                             "demo_agent" in body and "non_agentic_baseline" in body))
        results.append(check("dashboard draws SVG charts",
                             page.locator("#benchmark-body svg").count() >= 1))
        page.screenshot(path=str(shots / "06-benchmark-dashboard.png"), full_page=True)

        # 5c. Replay view plays a finished run back with its watermark.
        page.click('[data-view="replay"]')
        page.wait_for_timeout(800)
        options = page.locator("#replay-run option").count()
        results.append(check("replay lists finished runs", options >= 1,
                             f"{options} option(s)"))
        if options >= 1:
            page.select_option("#replay-run", index=0)
            page.click("#btn-replay")
            page.wait_for_selector("#replay-watermark", state="visible", timeout=15000)
            results.append(check("REPLAY watermark visible",
                                 page.locator("#replay-watermark").is_visible()))
            page.screenshot(path=str(shots / "07-replay.png"))
        browser.close()

    passed = sum(results)
    print(f"\nUI walkthrough: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
