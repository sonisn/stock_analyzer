"""The dashboard page: layout, styling and the client-side behaviour.

Split from the CLI so the generator stays about *what* is shown and this
file about *how*. The data is embedded as JSON and every interaction —
sorting, the ticker drill-down, the theme toggle — runs in the browser,
which is what lets the page be a single file that fetches nothing.

Light and dark are both designed rather than flipped: the dark steps come
from the same ramps, chosen for the dark surface. Charts are inline SVG,
so there is no library to vendor and nothing to load.
"""

from __future__ import annotations

from typing import Any

from .serialization import dumps_compact

CSS = """
:root{color-scheme:light;
 --bg:#f7f7f6;--surface:#fcfcfb;--line:#e3e2de;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7975;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0e6432;--bad:#9c1010;--warn:#8a5a00;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --bg:#111110;--surface:#1a1a19;--line:#33322f;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;}}
:root[data-theme="dark"]{color-scheme:dark;
 --bg:#111110;--surface:#1a1a19;--line:#33322f;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
.sub{color:var(--ink3);font-size:13px}
button.theme{margin-left:auto;background:var(--surface);color:var(--ink2);border:1px solid var(--line);
 border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:24px;font-weight:600;margin-top:4px;letter-spacing:-.02em}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:20px}
h2{font-size:15px;margin:0 0 4px;letter-spacing:-.01em}
h3{font-size:13px;margin:18px 0 6px;color:var(--ink2);font-weight:600}
.note{font-size:12.5px;color:var(--ink3);margin:0 0 14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-weight:500;color:var(--ink3);font-size:11.5px;text-transform:uppercase;
 letter-spacing:.04em;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line)}
tbody tr{cursor:pointer}
tbody tr:hover{background:color-mix(in srgb,var(--s1) 7%,transparent)}
tbody tr.on{background:color-mix(in srgb,var(--s1) 12%,transparent)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--good)}.neg{color:var(--bad)}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11.5px;border:1px solid var(--line)}
.pill.hold{color:var(--ink2)}.pill.trim{color:var(--warn);border-color:currentColor}
.pill.sell{color:var(--bad);border-color:currentColor}
.dim{color:var(--ink3)}
.flex{display:flex;gap:20px;flex-wrap:wrap}
.flex>*{flex:1 1 320px;min-width:0}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin-bottom:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:baseline}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--line);stroke-width:1}
.ax{fill:var(--ink3);font-size:10.5px}
.lbl{fill:var(--ink2);font-size:11px}
#detail,#why{display:none}
#detail.show,#why.show{display:block}
.viewtext{font-size:13px;color:var(--ink2);line-height:1.6;margin-top:10px;
 border-left:2px solid var(--s1);padding-left:12px}
.bar{height:7px;border-radius:4px;background:var(--s1)}
@media (max-width:640px){.hide-s{display:none}}
"""

JS = """
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const money = v => (v<0?'-$':'$')+Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0});
const pct = v => (v>=0?'+':'')+v.toFixed(1)+'%';
const cls = v => v>=0?'pos':'neg';
const dash = '<span class="dim">\u2014</span>';

/* theme */
const T='dash-theme';
function theme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem(T,t)}catch(e){}
  $('#tbtn').textContent = t==='dark'?'Light':'Dark';}
try{const s=localStorage.getItem(T); if(s) theme(s);}catch(e){}
$('#tbtn').onclick=()=>theme(document.documentElement.dataset.theme==='dark'?'light':'dark');

/* holdings table */
function verdictPill(v){if(!v) return '<span class="dim">—</span>';
  const k=String(v).toLowerCase();return `<span class="pill ${k}">${esc(v)}</span>`}
$('#holdings').innerHTML = D.holdings.map(h=>`
 <tr data-t="${h.ticker}">
  <td><b>${esc(h.ticker)}</b></td>
  <td>${verdictPill(h.verdict)} <span class="dim">${h.conf?h.conf+'/10':''}</span></td>
  <td class="num">${h.value===null?dash:money(h.value)}</td>
  <td class="num">${h.pl===null?dash:`<span class="${cls(h.pl)}">${pct(h.pl)}</span>`}</td>
  <td class="num hide-s">${h.book===null?'<span class="dim">—</span>':`<span class="${cls(h.book)}">${pct(h.book)}</span>`}</td>
  <td class="num hide-s">${h.calls?`${h.calls} <span class="dim">(${h.free} free)</span>`:'<span class="dim">—</span>'}</td>
 </tr>`).join('');

document.querySelectorAll('#holdings tr').forEach(tr=>tr.onclick=()=>show(tr.dataset.t));

/* ticker drill-down: confidence over runs */
function show(t){
  document.querySelectorAll('#holdings tr').forEach(r=>r.classList.toggle('on',r.dataset.t===t));
  const h=D.history[t]||[]; const view=D.views[t];
  $('#detail').classList.add('show');
  $('#dt').textContent=t;
  $('#dsum').innerHTML = h.length
    ? `${h.length} review${h.length>1?'s':''} on record · latest <b>${esc(h[h.length-1].v)}</b> at ${h[h.length-1].c}/10 on ${h[h.length-1].d}`
    : 'No review history.';
  $('#chart').innerHTML = h.length>1 ? spark(h) : '<p class="note">Not enough history to plot.</p>';
  $('#dview').innerHTML = view ? `<div class="viewtext">${esc(view)}</div>` : '';
  $('#detail').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function spark(h){
  const W=680,H=170,P={l:34,r:14,t:14,b:26};
  const x=i=>P.l+i*(W-P.l-P.r)/Math.max(1,h.length-1);
  const y=v=>P.t+(10-v)*(H-P.t-P.b)/10;
  let g='';
  for(let v=0;v<=10;v+=2) g+=`<line class="gl" x1="${P.l}" y1="${y(v)}" x2="${W-P.r}" y2="${y(v)}"/>
    <text class="ax" x="${P.l-7}" y="${y(v)+3.5}" text-anchor="end">${v}</text>`;
  const pts=h.map((d,i)=>`${x(i)},${y(d.c)}`).join(' ');
  const dots=h.map((d,i)=>{
    const c=d.v==='SELL'?'var(--bad)':d.v==='TRIM'?'var(--s2)':'var(--s1)';
    return `<circle cx="${x(i)}" cy="${y(d.c)}" r="4.5" fill="${c}" stroke="var(--surface)" stroke-width="2">
      <title>${d.d} — ${d.v} ${d.c}/10 (run ${d.run})</title></circle>`}).join('');
  const first=h[0].d.slice(5), last=h[h.length-1].d.slice(5);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reviewer confidence over time">
    ${g}<polyline points="${pts}" fill="none" stroke="var(--s1)" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>${dots}
    <text class="ax" x="${P.l}" y="${H-8}">${first}</text>
    <text class="ax" x="${W-P.r}" y="${H-8}" text-anchor="end">${last}</text></svg>`;
}

/* suggestions, graded against the price record */
const num = v => v===null||v===undefined ? '<span class="dim">—</span>'
  : `<span class="${v>=0?'pos':'neg'}">${(v>=0?'+':'')+v.toFixed(1)}%</span>`;
const vpill = v => v==='good call' ? '<span class="pill" style="color:var(--good);border-color:currentColor">good call</span>'
  : v==='missed' ? '<span class="pill" style="color:var(--bad);border-color:currentColor">missed</span>'
  : v==='too early' ? '<span class="pill dim">too early</span>' : '<span class="dim">—</span>';
$('#sugg').innerHTML = D.suggestions.map(s=>`<tr data-t="${s.ticker}">
  <td class="dim">${esc(s.d)}${s.run?`<div style="font-size:11px">run ${s.run}</div>`:''}</td>
  <td><b>${esc(s.ticker)}</b><div class="dim" style="font-size:11px">${esc(s.action)}</div></td>
  <td class="num">${num(s.ret)}</td>
  <td class="num hide-s">${num(s.spy)}</td>
  <td class="hide-s">${s.swap?`<b>${esc(s.swap)}</b> ${num(s.swap_pct)}`:'<span class="dim">—</span>'}</td>
  <td class="num">${num(s.edge)}</td>
  <td>${vpill(s.verdict)}</td>
  <td class="dim hide-s">${esc(s.acted)}</td></tr>`).join('');

document.querySelectorAll('#sugg tr').forEach(tr=>tr.onclick=()=>why(tr.dataset.t));

/* why a ticker was advised: scenarios, bull, bear, catalysts */
function why(t){
  const r = D.reasoning[t];
  const rev = D.reviews[t];
  $('#why').classList.add('show');
  $('#wt').textContent = t;
  if(!r && !rev){ $('#wbody').innerHTML =
    '<p class="note">No stored reasoning for this ticker — it came from a '+
    'deterministic rule in the daily email rather than from the ranker.</p>'; return; }
  let h='';
  if(r && r.scenarios && r.scenarios.length){
    const ev = r.scenarios.reduce((a,s)=>a + s.prob/100*s.target, 0);
    h += `<h3>Scenarios <span class="dim" style="font-weight:400">· expected `+
         `${ev>=0?'+':''}${ev.toFixed(1)}%/yr</span></h3>` + scen(r.scenarios);
  }
  if(r && r.bull) h += `<h3>Bull case</h3><div class="viewtext">${esc(r.bull)}</div>`;
  if(r && r.bear) h += `<h3>Bear case (red team)</h3>`+
    `<div class="viewtext" style="border-color:var(--bad)">${esc(r.bear)}</div>`;
  if(r && r.catalysts && r.catalysts.length){
    h += '<h3>Catalysts</h3><table><thead><tr><th>Event</th><th>When</th>'+
         '<th>Direction</th><th>Impact</th></tr></thead><tbody>'+
      r.catalysts.map(c=>`<tr><td>${esc(c.event)}</td>`+
        `<td class="dim">${esc(c.when||'date TBD')}</td>`+
        `<td>${esc(c.direction)}</td><td class="dim">${esc(c.impact)}</td></tr>`).join('')+
      '</tbody></table>';
  }
  if(rev) h += `<h3>Reviewer's note</h3><div class="viewtext">${esc(rev)}</div>`;
  $('#wbody').innerHTML = h;
  $('#why').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function scen(rows){
  const W=660,rowH=34,P={l:58,r:96,t:6};
  const H=P.t+rows.length*rowH+8;
  const lo=Math.min(-5,...rows.map(r=>r.target)), hi=Math.max(5,...rows.map(r=>r.target));
  const x=v=>P.l+(v-lo)/(hi-lo)*(W-P.l-P.r);
  const col={bull:'var(--good)',base:'var(--ink3)',bear:'var(--bad)'};
  let g=`<line class="gl" x1="${x(0)}" y1="${P.t}" x2="${x(0)}" y2="${H-8}"
     stroke="var(--ink3)" stroke-dasharray="3 3"/>`;
  rows.forEach((r,i)=>{
    const cy=P.t+i*rowH+rowH/2, c=col[r.label]||'var(--s1)';
    const x0=Math.min(x(0),x(r.target)), w=Math.max(2,Math.abs(x(r.target)-x(0)));
    g+=`<text class="lbl" x="0" y="${cy+4}">${r.label}</text>
      <rect x="${x0}" y="${cy-8}" width="${w}" height="16" rx="4" fill="${c}"
        opacity="${0.25+0.55*r.prob/100}"><title>${r.label}: ${r.prob}% chance of `+
      `${r.target>=0?'+':''}${r.target}%</title></rect>
      <text class="lbl" x="${W-P.r+10}" y="${cy+4}">${r.prob}% · `+
      `${r.target>=0?'+':''}${r.target}%</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scenario probabilities and targets">${g}</svg>`;
}

/* runs */
$('#runs').innerHTML = D.runs.map(r=>`<tr>
  <td class="dim">#${r.id}</td><td>${esc(r.kind)}</td><td class="dim">${esc(r.d)}</td>
  <td class="num">${r.universe_size||'—'}</td><td class="num">${r.survivors||'—'}</td>
  <td class="num">${r.picks||'—'}</td></tr>`).join('');

if(D.holdings.length) show(D.holdings[0].ticker);
"""


def render_page(data: dict[str, Any]) -> str:
    """The whole page, with `data` embedded."""
    holdings = data["holdings"]
    total = sum(h["value"] or 0 for h in holdings)
    flagged = sum(1 for h in holdings if h.get("verdict") and h["verdict"] != "HOLD")
    withcalls = sum(1 for h in holdings if h.get("calls"))
    rec = data["record"]
    stale = "" if data.get("holdings_ok") else (
        '<div class="card" style="border-color:var(--bad)"><b>Holdings unavailable.</b> '
        "The brokerage could not be reached when this page was generated, so the table "
        "below is empty. Everything else is from the database and is unaffected.</div>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Dashboard</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>Portfolio</h1>
 <span class="sub">run #{data['latest_run']} · generated {data['generated']}</span>
 <button class="theme" id="tbtn">Dark</button></header>
{stale}
<div class="tiles">
 <div class="tile"><div class="k">Holdings value</div><div class="v">${total:,.0f}</div></div>
 <div class="tile"><div class="k">Positions</div><div class="v">{len(holdings)}</div></div>
 <div class="tile"><div class="k">Needing a decision</div><div class="v">{flagged}</div></div>
 <div class="tile"><div class="k">With calls written</div><div class="v">{withcalls}</div></div>
</div>

<div class="card">
 <h2>Holdings</h2>
 <p class="note">Click a row for its full review history. “Book” is the SEC-filed
  contracted order book year over year — a dash means the company doesn’t tag it,
  not that it has none.</p>
 <table><thead><tr><th>Ticker</th><th>Verdict</th><th class="num">Value</th>
  <th class="num">P/L</th><th class="num hide-s">Book YoY</th>
  <th class="num hide-s">Calls</th></tr></thead><tbody id="holdings"></tbody></table>
</div>

<div class="card" id="detail">
 <h2><span id="dt"></span> — review history</h2>
 <p class="note" id="dsum"></p>
 <div class="legend"><span><i style="background:var(--s1)"></i>HOLD</span>
  <span><i style="background:var(--s2)"></i>TRIM</span>
  <span><i style="background:var(--bad)"></i>SELL</span></div>
 <div id="chart"></div><div id="dview"></div>
</div>

<div class="card"><h2>Suggestions, graded</h2>
 <p class="note">Every piece of advice, measured against the price record
  ({rec['rows']:,} closes on {rec['tickers']} tickers, {rec['first']} to {rec['last']}).
  <b>A sale is scored against what replaced it</b> — edge is the replacement’s return
  minus the sold stock’s, so positive means the switch was worth making. Anything
  under 30 days old reads <i>too early</i> rather than inventing a verdict.</p>
 <table><thead><tr><th>Suggested</th><th>Ticker</th><th class="num">Return</th>
  <th class="num hide-s">SPY</th><th class="hide-s">Replaced by</th>
  <th class="num">Edge</th><th>Verdict</th><th class="hide-s">Acted</th></tr></thead>
 <tbody id="sugg"></tbody></table>
 <p class="note" style="margin-top:12px">Click any row for the reasoning behind it.</p></div>

<div class="card" id="why">
 <h2><span id="wt"></span> — why</h2>
 <div id="wbody"></div>
</div>

<div class="card"><h2>Recent runs</h2>
 <p class="note">Pipeline history.</p>
 <table><thead><tr><th>Run</th><th>Kind</th><th>Date</th><th class="num">Universe</th>
  <th class="num">Surv.</th><th class="num">Picks</th></tr></thead>
 <tbody id="runs"></tbody></table></div>

</div><script>{JS.replace("__DATA__", dumps_compact(data))}</script></body></html>"""
