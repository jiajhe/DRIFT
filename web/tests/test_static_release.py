"""Exercise the published ZIP from a plain HTTP server with the GPU API disconnected."""

from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import tempfile
import threading
import zipfile

import pytest

REPORTS = Path(__file__).resolve().parents[2] / "reports"


@contextmanager
def static_server(folder):
    class QuietFiles(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietFiles, directory=folder)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_judge_zip_and_screenshots():
    pw = pytest.importorskip("playwright.sync_api")
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(REPORTS / "marketflow-static.zip") as package:
        assert set(package.namelist()) == {
            "index.html", "app.js", "style.css", "config.js", "archive.js", "model-card.json", "historical-demo.json",
            "multiday.js", "multiday-replay.json"
        }
        package.extractall(folder)
        archive = json.loads(package.read("historical-demo.json"))
        evaluation = json.loads(package.read("model-card.json"))["evaluation"]
        assert (archive["gate"], archive["checkpoint_sha256"]) == (evaluation["gate"], evaluation["checkpoint_sha256"])
        with static_server(folder) as url, pw.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=2)
            failures = []
            page.on("pageerror", lambda error: failures.append(str(error)))
            page.route("**/api/**", lambda route: route.abort("connectionrefused"))
            page.goto(url)
            page.locator("#content").wait_for()
            check = pw.expect
            check(page.locator("#source-mode")).to_contain_text("Archived inference")
            check(page.locator("#footer-gate")).to_have_text("Grounding test: " + evaluation["gate"])
            check(page.locator("#target-metric")).to_have_text("AAPL")
            check(page.locator("#senders .row")).to_have_count(5)
            check(page.locator("#date option")).to_have_count(12)
            for task in ["ATTRIBUTION_COT", "REGIME_ROLE_COT", "PAIRWISE_FORECAST_COT"]:
                page.select_option("#task", task, force=True)
                if task == "PAIRWISE_FORECAST_COT":
                    check(page.locator("#pair-symbols option")).to_have_count(7)
                    page.fill("#pair", "NVDA")
                page.click("#ask")
                check(page.locator("#answer dd").first).to_be_visible()
                check(page.locator("#answer small")).to_contain_text("archived output")
            page.locator("#senders button").nth(1).click()
            page.click("#mask")
            check(page.locator("#task")).to_have_value("COUNTERFACTUAL_COT")
            check(page.locator("#mask-result")).to_contain_text("Δ")
            check(page.locator("#answer dd")).to_have_count(3)
            sender = page.locator("#answer dt").filter(has_text="Queried sender").locator("xpath=following-sibling::dd[1]").inner_text()
            check(page.locator("#mask")).to_contain_text(sender)
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=REPORTS / "v2_demo.png", full_page=True)
            page.get_by_role("link", name="Replay", exact=True).click()
            page.click("#next")
            check(page.locator("#date")).to_have_value("2024-01-03")
            page.click("#prev")
            check(page.locator("#date")).to_have_value("2024-01-02")
            page.select_option("#date", "2025-12-29")
            page.fill("#ticker", "MSFT")
            page.click("#load")
            check(page.locator("#target-metric")).to_have_text("MSFT")
            page.get_by_role("link", name="Model card").click()
            check(page.get_by_role("heading", name="Native DRIFT and winner-logit scores")).to_be_visible()
            check(page.locator("#model-content")).to_contain_text("Grounding test: " + evaluation["gate"])
            page.evaluate("window.scrollTo(0,0)")
            margin_note = page.get_by_text("Qwen compares its A and B logits", exact=False).bounding_box()
            page.set_viewport_size({"width":1440, "height":round(margin_note["y"] + margin_note["height"]) + 28})
            page.screenshot(path=REPORTS / "v2_model_card.png")
            counts = page.evaluate("""async () => {
              const {staticRequest: get} = await import('./archive.js');
              const results = {days:0, answers:0, masks:0};
              for (const date of (await get('catalog')).dates) {
                const {targets} = await get('catalog?date='+date);
                for (const target of targets) {
                  const row = {date,target}, graph = await get('snapshot?'+new URLSearchParams(row));
                  if (graph.target !== target || graph.date !== date || graph.edges.length !== 5) throw Error('Snapshot key');
                  results.days++;
                  for (const task of ['ATTRIBUTION_COT','REGIME_ROLE_COT','COUNTERFACTUAL_COT','PAIRWISE_FORECAST_COT']) {
                    const opts = task.startsWith('COUNTER') ? graph.edges.map(e=>({sender:e.source})) :
                      task.startsWith('PAIR') ? targets.filter(s=>s!==target).map(other=>({other})) : [{}];
                    for (const opt of opts) {
                      const ans=await get('copilot',{...row,task,...opt});
                      if (!ans.cached || !ans.reasoning || !ans.answer || ans.task!==task || ans.sample_id!=='SP500|'+date+'|'+target) throw Error('Answer key');
                      results.answers++;
                      if (opt.sender) {
                        const m=await get('intervene',{...row,...opt});
                        if (m.sender!==opt.sender || m.target!==target || m.masked_edge!==0 || Math.abs(m.delta-(m.intervened_score-m.original_score))>1e-10) throw Error('Masked edge');
                        results.masks++;
                      }
                    }
                  }
                }
              }
              return results;
            }""")
            assert counts == {"days": 96, "answers": 1344, "masks": 480}
            assert not failures, failures
            # A static host may send a JSON 404 or an HTML error page for unmatched API paths.
            phone = browser.new_page(viewport={"width": 390, "height": 844})
            phone.route("**/api/**", lambda r: r.fulfill(status=404, json={"detail": "Not Found"}))
            phone.goto(url)
            check(phone.locator("#target-metric")).to_have_text("AAPL")
            phone.click("#mask")
            check(phone.locator("#answer small")).to_contain_text("archived output")
            browser.close()


def test_multiday_static_replay():
    """Use the actual release ZIP and both market calendars, then disconnect even the static server."""
    pw = pytest.importorskip("playwright.sync_api")
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(REPORTS / "marketflow-static.zip") as z:
        archive = json.loads(z.read("multiday-replay.json"))
        z.extractall(folder)
        with static_server(folder) as url, pw.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width":1440,"height":1100},device_scale_factor=2)
            page = context.new_page()
            errors, api = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            def disconnected(route):
                api.append(route.request.url)
                route.abort("connectionrefused")
            context.route("**/api/**", disconnected)
            page.goto(url+"#multiday")
            page.locator("#md-chart polyline").first.wait_for()
            assert not api
            context.set_offline(True)
            page.click("#md-details summary")
            page.check("#md-precision")

            for market, group in archive["markets"].items():
                page.click(f'#md-markets button[data-market="{market}"]')
                cases = group["cases"]
                assert page.locator("#md-case option").count() == len(cases)
                # Read back every table value, chart coordinate, classification and fifth-day readout.
                page.evaluate("""cases => {
                  const id = s => document.getElementById(s);
                  for (const [index,c] of cases.entries()) {
                    id('md-case').value=String(index);
                    id('md-case').dispatchEvent(new Event('change'));
                    for (const mode of ['today','five','restored']) {
                      document.querySelector('[data-mode="'+mode+'"]').click();
                      const [a,b] = [c.sender_a,c.sender_b].map(s=>c.trajectories[s][mode]);
                      const rows = Array.from(id('md-table').tBodies[0].rows);
                      if(rows.length!==5) throw Error('Wrong day count');
                      rows.forEach((row,i)=>{
                        const [date,...numbers]=Array.from(row.cells,x=>x.textContent);
                        const expected=[c.base_scores[i],a.scores[i],a.delta[i],b.scores[i],b.delta[i]];
                        if(date!==c.window_dates[i] || numbers.some((v,j)=>Number(v)!==expected[j])) throw Error('Wrong table data');
                      });
                      const ax=Math.abs(a.delta[4]), bx=Math.abs(b.delta[4]);
                      const winner=Math.abs(ax-bx)<0.05*Math.max(ax,bx)?'Near Tie':ax>bx?c.sender_a:c.sender_b;
                      if(id('md-winner').textContent!==winner ||
                         Number(id('md-delta-a').title)!==a.delta[4] || Number(id('md-delta-b').title)!==b.delta[4]) throw Error('Wrong day-five readout');
                      const svg=id('md-chart'), lines=Array.from(svg.querySelectorAll('polyline'));
                      if(lines.length!==2 || !svg.querySelector('.md-zero')) throw Error('Missing chart');
                      lines.forEach((line,j)=>{
                        const points=line.points, g=line.parentElement, vals=(j?b:a).delta;
                        const dots=g.querySelectorAll('circle');
                        if(points.numberOfItems!==5 || dots.length!==5 || dots[4].dataset.session!=='5') throw Error('Missing endpoint');
                        if(g.dataset.channel!==(j?c.sender_b:c.sender_a)) throw Error('Sender mismatch');
                        for(let i=0;i<5;i++){
                          const pt=points.getItem(i), dot=dots[i];
                          const offset=svg.querySelector('.md-zero').y1.baseVal.value - pt.y;
                          if (vals[i]===0 ? Math.abs(offset)>0.0001 : Math.sign(offset)!==Math.sign(vals[i])) throw Error('Wrong side of zero');
                          if(Math.abs(pt.y-Number(dot.getAttribute('cy')))>0.0001 ||
                             Number(dot.querySelector('title').textContent.split(' · ').at(-1))!==vals[i]) throw Error('Chart datum mismatch');
                        }
                      });
                    }
                  }
                }""", cases)
                target = "GOOGL" if market=="SP500" else "SH600519"
                pos = next(i for i,c in enumerate(cases) if c["target"]==target)
                page.select_option("#md-case", str(pos))
                page.click('#md-modes [data-mode="restored"]')
                page.uncheck("#md-precision")
                pw.expect(page.locator("#md-chart")).to_contain_text("RESTORED")
                pw.expect(page.locator("#md-mechanism")).to_contain_text("open again")
                page.evaluate("window.scrollTo(0,0)")
                page.screenshot(path=REPORTS / f"v2_multiday_{market.lower()}.png", full_page=True)
                page.check("#md-precision")

            page.set_viewport_size({"width":390,"height":844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not api and not errors
            context.set_offline(False)
            page.get_by_role("link",name="Flow Explorer",exact=True).click()
            page.locator("#content").wait_for()
            pw.expect(page.locator("#task option")).to_have_count(4)
            browser.close()
