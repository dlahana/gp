"""HTML for the rating and approval pages (no build step, no templates dir)."""

from __future__ import annotations

import html
import json

BASE_CSS = """
:root { --bg:#f7f5ef; --fg:#1d1d1b; --muted:#6b6a64; --card:#fff; --line:#e3e0d6; --accent:#c8e05a; --ink:#1f3d2b; }
@media (prefers-color-scheme: dark) { :root { --bg:#151614; --fg:#ecebe6; --muted:#a2a19a; --card:#1f201d; --line:#33342f; --accent:#b8d043; --ink:#d9f07a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:560px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--muted); font-size:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin-top:16px; }
button { font:inherit; cursor:pointer; border-radius:10px; border:1px solid var(--line); background:var(--card); color:var(--fg); padding:12px 14px; }
button.primary { background:var(--accent); color:#1d1d1b; border-color:transparent; font-weight:600; }
button:disabled { opacity:.5; cursor:default; }
textarea { width:100%; font:inherit; padding:10px; border-radius:10px; border:1px solid var(--line); background:var(--bg); color:var(--fg); min-height:110px; }
"""

RATE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Nickname Court</title>
<style>__CSS__
.vs { font-size:15px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); text-align:center; margin:8px 0 14px; }
.opts { display:grid; gap:10px; } .opts button { font-size:22px; font-weight:650; text-align:left; padding:16px; }
.opts button:hover { border-color:var(--ink); }
.none { margin-top:14px; width:100%; color:var(--muted); }
.count { text-align:right; } </style></head><body><main>
<h1>🎾 Nickname Court</h1>
<p class="muted">Two tennis players, a few mashed-up nicknames. Tap the funniest, stupidest one.
Your picks teach the bot what's funny.</p>
<div class="card"><div class="vs" id="vs">loading…</div><div class="opts" id="opts"></div>
<button class="none" id="none">None of these are funny</button></div>
<p class="muted count" id="count"></p>
</main><script>
const code = __CODE__;
let rater = '';
try { rater = localStorage.getItem('rater') || ''; if (!rater) { rater = Math.random().toString(36).slice(2, 12); localStorage.setItem('rater', rater); } } catch (e) {}
let round = null, busy = false;
function render(r) {
  round = r;
  document.getElementById('vs').textContent = r.name_a + '  ×  ' + r.name_b;
  const box = document.getElementById('opts'); box.replaceChildren();
  r.shown.forEach((nick, i) => { const b = document.createElement('button');
    b.textContent = nick.charAt(0).toUpperCase() + nick.slice(1); b.onclick = () => vote(i); box.appendChild(b); });
  document.getElementById('count').textContent = r.my_votes ? r.my_votes + ' picks so far, thank you' : '';
}
async function load() { const r = await fetch('/api/round/' + code + '?rater=' + encodeURIComponent(rater)); if (!r.ok) { document.getElementById('vs').textContent = 'This link is not valid.'; return; } render(await r.json()); }
async function vote(i) { if (busy || !round) return; busy = true;
  try { const r = await fetch('/api/vote/' + code, {method:'POST', headers:{'content-type':'application/json'},
      body: JSON.stringify({round_id: round.round_id, chosen: i, rater})}); render(await r.json()); } finally { busy = false; } }
document.getElementById('none').onclick = () => vote(null);
load();
</script></body></html>"""

APPROVE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Approve Picks</title>
<style>__CSS__
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; } .chips span { display:inline-block; border:1px solid var(--line);
border-radius:999px; padding:2px 10px; margin:3px 4px 0 0; font-size:14px; } .chips .used { background:var(--accent); color:#1d1d1b; border-color:transparent; }
.len { font-size:13px; color:var(--muted); text-align:right; } .status { font-weight:600; margin-top:12px; } input { width:100%; font:inherit; padding:10px;
border-radius:10px; border:1px solid var(--line); background:var(--bg); color:var(--fg); } </style></head><body><main>
<h1>__KIND__ for __DAY__ (__TOUR__)</h1><p class="muted">Status: <b id="st">__STATUS__</b> · written by __SOURCE__.
Edit freely; nothing is posted until you tap Approve.</p>
<div id="tweets"></div>
<div class="card"><b>Matches and nickname options</b>__MATCHES__</div>__FACTS__
<div class="card"><input id="dir" placeholder="Direction for a rewrite (optional), e.g. more puns, use 'naderer'">
<div class="row"><button id="regen">Rewrite</button><button id="reject">Reject</button>
<button class="primary" id="approve">Approve &amp; post</button></div><div class="status" id="msg"></div></div>
</main><script>
const id = __ID__, token = __TOKEN__; let tweets = __TWEETS__; const locked = __LOCKED__;
function draw() { const box = document.getElementById('tweets'); box.replaceChildren();
  tweets.forEach((t, i) => { const c = document.createElement('div'); c.className = 'card';
    const ta = document.createElement('textarea'); ta.value = t; const n = document.createElement('div'); n.className = 'len';
    const upd = () => { tweets[i] = ta.value; n.textContent = ta.value.length + ' / 280'; n.style.color = ta.value.length > 280 ? '#d33' : ''; };
    ta.oninput = upd; ta.disabled = locked; upd(); c.append(ta, n); box.appendChild(c); }); }
async function call(path, body) { const msg = document.getElementById('msg'); msg.textContent = 'working…';
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  const r = await fetch('/api/drafts/' + id + '/' + path + '?t=' + encodeURIComponent(token), {method:'POST',
    headers:{'content-type':'application/json'}, body: JSON.stringify(body || {})});
  const j = await r.json(); msg.textContent = j.message || (r.ok ? 'done' : 'error');
  if (j.tweets) { tweets = j.tweets; draw(); } if (j.status) document.getElementById('st').textContent = j.status;
  if (!j.locked) document.querySelectorAll('button').forEach(b => b.disabled = false); }
document.getElementById('approve').onclick = () => { if (confirm('Post ' + tweets.length + ' tweet(s) to X now?')) call('approve', {tweets}); };
document.getElementById('regen').onclick = () => call('regenerate', {direction: document.getElementById('dir').value});
document.getElementById('reject').onclick = () => call('reject');
draw(); if (locked) document.querySelectorAll('button').forEach(b => b.disabled = true);
</script></body></html>"""


def _js(value) -> str:
    """JSON for embedding in a <script> block."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def rate_page(code: str) -> str:
    return RATE_PAGE.replace("__CSS__", BASE_CSS).replace("__CODE__", _js(code))


def approve_page(draft: dict) -> str:
    rows = []
    used_list = draft["nicknames_used"] or []
    for i, m in enumerate(draft["matches"]):
        used = used_list[i] if i < len(used_list) else ""
        p = max(m["p1"], 1 - m["p1"])
        fav = m["player1"] if m["p1"] >= 0.5 else m["player2"]
        chips = "".join(f'<span class="{"used" if n == used else ""}">{html.escape(n)}</span>' for n in m["nicknames"])
        rows.append(f'<div style="margin-top:12px"><div>{html.escape(m["player1"])} vs {html.escape(m["player2"])} · '
                    f'{html.escape(fav)} {p:.0%}</div><div class="chips">{chips}</div></div>')
    locked = draft["status"] != "pending"
    facts = ""
    if draft.get("facts"):
        from ..insights import render_text
        facts = (f'<div class="card"><b>The numbers behind it</b><pre style="white-space:pre-wrap;font-size:13px">'
                 f'{html.escape(render_text(draft["facts"]))}</pre></div>')
    return (APPROVE_PAGE.replace("__CSS__", BASE_CSS)
            .replace("__KIND__", "Match insights" if draft["kind"] == "insights" else "Picks")
            .replace("__DAY__", html.escape(draft["day"])).replace("__TOUR__", html.escape(draft["tour"].upper()))
            .replace("__STATUS__", html.escape(draft["status"])).replace("__SOURCE__", html.escape(draft["source"]))
            .replace("__MATCHES__", "".join(rows)).replace("__FACTS__", facts).replace("__ID__", _js(draft["id"]))
            .replace("__TOKEN__", _js(draft["token"])).replace("__TWEETS__", _js(draft["tweets"]))
            .replace("__LOCKED__", _js(locked)))
