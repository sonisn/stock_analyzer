"""The dashboard's Market leaders tab: a MarketSurge-style screen.

Laid out the way that product works, without its name or marks: the
market's direction across the top; a stock list on the left with ratings
in columns and switchable views (buy zone, leaders, our picks, holdings,
groups); and on the right the selected stock's rating badges, a daily
chart and this app's own data about it.

The chart is three stacked panes on a shared date axis, each with its own
scale rather than one pane with two: price (high-low-close bars, 50- and
200-day averages, the base's pivot and the 5% buy zone above it), the RS
line (price relative to SPY) and volume (with its 50-day average). A
crosshair reads all three at once. Colours come from the page's validated
series tokens (--s1 blue, --s2 orange, --s3 green); up and down days are
blue and orange, with a legend, never colour alone.

Data: reporting/leaders.py. Styling tokens: dashboard_page.CSS.
"""

from __future__ import annotations

CSS = """
.pulse{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px}
.pulse .idx{font-size:12.5px;color:var(--ink2)}
.pulse .idx b{color:var(--ink)}
.mkt{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:99px;font-weight:600;
 font-size:13px;border:1px solid currentColor}
.mkt.up{color:var(--good)}.mkt.press{color:var(--warn)}.mkt.down{color:var(--bad)}
.strip{display:flex;gap:2px;margin-top:12px}
.strip i{flex:1;height:8px;border-radius:2px}
.ms{display:grid;grid-template-columns:minmax(300px,380px) minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:900px){.ms{grid-template-columns:1fr}}
.ms-list{padding:12px}
.seg{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.seg button{background:var(--bg);color:var(--ink2);border:1px solid var(--line);border-radius:7px;
 padding:4px 9px;font:inherit;font-size:12.5px;cursor:pointer}
.seg button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.seg button .n{opacity:.65;margin-left:4px}
.ms-scroll{max-height:72vh;overflow:auto;margin:0 -12px;padding:0 12px}
.ms-scroll table{font-size:12.5px}
.ms-scroll th{position:sticky;top:0;background:var(--surface);z-index:1}
.ms-scroll td{padding:5px 6px}.ms-scroll th{padding:5px 6px}
.sig{display:inline-block;min-width:16px;padding:0 4px;margin-right:2px;border-radius:4px;font-size:10.5px;
 font-weight:600;text-align:center;border:1px solid var(--line);color:var(--ink2)}
.rate{font-variant-numeric:tabular-nums;font-weight:600}
.rate.lo{color:var(--ink3);font-weight:400}
.filter{width:100%;background:var(--bg);color:var(--ink);border:1px solid var(--line);
 border-radius:8px;padding:6px 10px;font:inherit;font-size:13px;margin-bottom:8px}
.dhead{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin-bottom:12px}
.dhead h2{font-size:22px;margin:0;letter-spacing:-.02em}
.dhead .px{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.badges{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:8px;margin-bottom:14px}
.badge{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--bg)}
.badge .k{font-size:10.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em}
.badge .v{font-size:22px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.badge .v small{font-size:12px;color:var(--ink3);font-weight:400}
.meter{height:4px;border-radius:2px;background:color-mix(in srgb,var(--s1) 18%,transparent);margin-top:4px}
.meter i{display:block;height:4px;border-radius:2px;background:var(--s1)}
.chartwrap{position:relative}
.table-scroll{overflow-x:auto}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);
 border-radius:8px;padding:7px 9px;font-size:12px;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.12);
 display:none;white-space:nowrap;z-index:2}
.tip b{font-variant-numeric:tabular-nums}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px 18px;margin-top:6px}
.fact{font-size:13px;border-top:1px solid var(--line);padding-top:7px}
.fact .k{font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em}
.key{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.key.bar{height:9px;width:3px;border:0;border-radius:1px}
details.how summary{cursor:pointer;color:var(--ink2);font-size:13px}
:root{--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300;--c7:#4a3aa7;--c8:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;--c7:#9085e9;--c8:#e66767}}
:root[data-theme="dark"]{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;--c7:#9085e9;--c8:#e66767}
.sstat{display:inline-flex;align-items:center;gap:5px;font-weight:600;font-size:12.5px;white-space:nowrap}
.sstat.Leading,.sstat.Uptrend{color:var(--good)}.sstat.Caution{color:var(--warn)}.sstat.Correction{color:var(--bad)}
tr.mine td:first-child{box-shadow:inset 3px 0 0 var(--s1)}
.minis{display:flex;gap:1px;min-width:90px}.minis i{flex:1;height:8px;border-radius:1px}
"""

HTML = """
<section class="tab" id="tab-leaders">
<div class="card" id="ms-pulse"></div>
<div class="card" id="ms-sectors"></div>
<div class="card" id="ms-groups"></div>
<div class="ms">
 <div class="card ms-list">
  <div class="seg" id="ms-seg" role="tablist" aria-label="Stock lists"></div>
  <input class="filter" id="ms-q" placeholder="Filter: ticker or industry" aria-label="Filter the list">
  <div class="ms-scroll"><table><thead id="ms-head"></thead><tbody id="ms-rows"></tbody></table></div>
  <p class="note" style="margin:8px 0 0">↑/↓ to move through the list. Signals:
   <span class="sig">H</span>held <span class="sig">P</span>our pick <span class="sig">B</span>book growing
   <span class="sig">I</span>insider buying <span class="sig">F</span>funds adding <span class="sig">S</span>earnings standout</p>
 </div>
 <div class="card" id="ms-detail"></div>
</div>
<div class="card" id="ms-map"></div>
<div class="card" id="ms-score"></div>
<div class="card"><details class="how"><summary>How these ratings are computed</summary>
 <p class="note" style="margin:10px 0 0">Reconstructed from the rules Investor’s Business Daily publishes, over
  every US-listed stock worth $2B+ — not IBD’s data. <b>RS</b>: 12-month price change with the latest quarter
  counted twice, ranked 1–99. <b>EPS</b>: the last two quarters’ EPS growth year over year and the 3-year
  annual rate, from SEC filings, ranked 1–99. <b>Group</b>: industry rank by median 6-month change
  (1 = strongest). <b>A/D</b>: 13 weeks of volume weighted by where each day closed in its range, A = heaviest
  buying. <b>Composite</b>: EPS and RS counted twice plus group, A/D and nearness to the 52-week high (IBD also
  counts sales, margins and ROE). <b>Bases</b>: flat base (≤15% deep, 5+ weeks) or cup (≤35%, 7+ weeks, with a
  handle when the right side ends in a shallow pullback in its upper half), after a 30% run-up; the pivot is the
  base’s high plus 10¢ and the buy zone runs 5% above it. <b>Market</b>: distribution days (index down 0.2%+ on
  higher volume, last 25 sessions) and follow-through days on SPY and QQQ. <b>Screen</b> is this app’s own
  discover score from its latest run. For a 3–5 year portfolio, read all of this as a sign of strength, not as
  a timing signal.</p></details></div>
</section>
"""

JS = r"""
/* ---- Market leaders: MarketSurge-style screen ---- */
(function(){
const L = D.ibd || {};
const root = $('#tab-leaders');
if(!L.rows){
  root.innerHTML = '<div class="card"><h2>Market leaders</h2><p class="note">No ratings yet. They are '+
    'computed each weekday at 6 AM New York time (<code>uv run ibd-ratings</code> by hand).</p></div>';
  return;
}
const HOLD = Object.fromEntries((D.holdings||[]).map(h=>[h.ticker,h]));
const fmt = (v,d=2) => v===null||v===undefined ? '—' : Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const sgn = (v,d=1) => v===null||v===undefined ? '—' : (v>=0?'+':'')+v.toFixed(d)+'%';
const fr = v => v===null||v===undefined ? dash : `<span class="${v>=0?'pos':'neg'}">${sgn(v*100)}</span>`;
const mktClass = s => /correction/i.test(s)?'down':/pressure/i.test(s)?'press':'up';
const mktIcon = s => ({up:'▲',press:'◆',down:'▼'})[mktClass(s)];
const rate = v => v===null||v===undefined ? dash : `<span class="rate ${v<50?'lo':''}">${v}</span>`;

/* market direction strip */
(function(){
  const m=L.market;
  let h = '<div class="pulse">';
  h += m ? `<span class="mkt ${mktClass(m.status)}"><span aria-hidden="true">${mktIcon(m.status)}</span>${esc(m.status)}</span>
    <span class="idx">${esc(m.detail)}</span>` : '<span class="dim">Market direction unavailable</span>';
  if(m) for(const [k,v] of Object.entries(m.indexes)) h += `<span class="idx"><b>${k}</b> ${v.distribution_days} dist. day${v.distribution_days===1?'':'s'}
    · ${(v.drawdown*100).toFixed(1)}% off high · ${v.above_50dma?'above':'below'} 50-day${v.follow_through?` · FTD ${v.follow_through}`:''}</span>`;
  h += `<span class="idx" style="margin-left:auto">as of the ${esc(L.as_of)} close · ${L.rated.toLocaleString()} stocks · ${L.groups_ranked} groups</span></div>`;
  const hist = L.market_history||[];
  if(hist.length>1) h += '<div class="strip" aria-label="Market direction, recent sessions">'+hist.map(x=>
    `<i title="${x.d}: ${esc(x.s)}" style="background:var(--${{up:'good',press:'warn',down:'bad'}[mktClass(x.s)]})"></i>`).join('')+'</div>';
  $('#ms-pulse').innerHTML = h;
})();

/* sector direction: Leading / Uptrend / Caution / Correction, with reasons */
const SICON = {Leading:'\u25b2', Uptrend:'\u25b2', Caution:'\u25c6', Correction:'\u25bc'};
const sstat = st => `<span class="sstat ${st}"><span aria-hidden="true">${SICON[st]||''}</span>${esc(st)}</span>`;
const SCOLOR = {Leading:'good', Uptrend:'good', Caution:'warn', Correction:'bad'};
(function(){
  const sv = L.sectors || {}, el = $('#ms-sectors');
  if(!sv.latest){ el.innerHTML = '<h2>Sectors</h2><p class="note">Appears after the next morning run.</p>'; return; }
  const mine = new Set(L.rows.filter(r=>r.held).flatMap(r=>[r.sec, r.ind==='Semiconductors'?'Semiconductors':null]).filter(Boolean));
  const rows = sv.latest.map(x=>{
    const hist = (sv.history[x.sector]||[]).slice(-30);
    const b = x.breadth, bb = x.breadth_before, arrow = b-bb>=0.03?'\u2191':b-bb<=-0.03?'\u2193':'';
    const etf = x.etf ? `${esc(x.etf)} ${x.etf_above_50===null?'':x.etf_above_50?'above':'below'} 50-day${x.etf_dist_days?` · ${x.etf_dist_days} dist.`:''}` : '\u2014';
    return `<tr class="${mine.has(x.sector)?'mine':''}"><td><b>${esc(x.sector)}</b>${mine.has(x.sector)?' <span class="pill">you hold</span>':''}</td>
      <td>${sstat(x.status)}</td><td class="num">${x.rank??'\u2014'}</td>
      <td class="num">${(x.median_six*100>=0?'+':'')+(x.median_six*100).toFixed(1)}%</td>
      <td class="num">${Math.round(b*100)}% ${arrow}</td><td class="num">${x.leaders}</td>
      <td class="dim hide-s">${etf}</td>
      <td class="hide-s"><div class="minis" title="Last ${hist.length} sessions">${hist.map(h=>`<i title="${h.d}: ${esc(h.s)}" style="background:var(--${SCOLOR[h.s]||'ink3'})"></i>`).join('')}</div></td>
      <td class="dim">${x.reasons?esc(x.reasons):''}</td></tr>`;}).join('');
  el.innerHTML = `<h2>Sectors: where the leadership is</h2>
    <p class="note">Checked every morning. <b>Caution</b>: the sector’s ETF lost its 50-day average, has 5+ distribution days,
    or fewer than half its stocks are above their 50-day and falling: a sign to stop adding new money there.
    <b>Correction</b>: the ETF is more than 2% below its 200-day, or under 30% of its stocks are above their 50-day: consider trimming its
    weakest names. <b>Leading</b>: healthy and a top-3 sector by 6-month change. Breadth is the share of its stocks above
    their 50-day average (arrow: change over two weeks).</p>
    <div class="table-scroll"><table><thead><tr><th>Sector</th><th>Status</th><th class="num">Rank</th><th class="num">6 months</th>
    <th class="num">Breadth</th><th class="num">Comp 90+</th><th class="hide-s">ETF</th><th class="hide-s">Last 30 sessions</th><th>Why</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
})();

/* a year of the strongest industry groups (equal-weight index of their stocks) against SPY */
let groupView = 'top';
function groupChart(){
  const gc = L.group_chart || {}, el = $('#ms-groups');
  if(!gc.groups || !gc.groups.length){ el.innerHTML = '<h2>Industry groups</h2><p class="note">Needs a year of stored prices.</p>'; return; }
  const shown = gc.groups.filter(g=> groupView==='top' ? g.top : g.held).slice(0,8);
  const nTop = gc.groups.filter(g=>g.top).length, nMine = gc.groups.filter(g=>g.held).length;
  const head = `<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px"><h2 style="margin:0">Industry groups over the last year</h2>
    <div class="seg" style="margin:0" role="tablist" aria-label="Which groups">
    <button data-g="top" class="${groupView==='top'?'on':''}" aria-selected="${groupView==='top'}">Top 5 now<span class="n">${nTop}</span></button>
    <button data-g="mine" class="${groupView==='mine'?'on':''}" aria-selected="${groupView==='mine'}">Your holdings’ groups<span class="n">${nMine}</span></button></div></div>
    <p class="note" style="margin-top:8px">Each line is an equal-weight index of the group’s rated stocks, rebased to 100 a year ago
    (gray: SPY on the same scale). A line pulling away from SPY is leadership; one rolling over is a group losing it.
    Top 5 is today’s group rank.</p>`;
  const W=Math.max(320, Math.round(el.clientWidth||760)-36), H=300, P={l:8,r:170,t:10,b:22};
  const n=gc.dates.length, x=i=>P.l+i*(W-P.l-P.r)/(n-1);
  const all=[...gc.spy, ...shown.flatMap(g=>g.series)].filter(v=>v!==null);
  const lo=Math.min(...all), hi=Math.max(...all), pad=(hi-lo)*0.05, y=v=>P.t+(hi+pad-v)/((hi-lo)+2*pad)*(H-P.t-P.b);
  let g='';
  for(const v of niceTicks(lo,hi,5)) g+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${W-P.r+6}" y="${y(v)+3.5}">${v}</text>`;
  let lastM=''; gc.dates.forEach((d,i)=>{const m=d.slice(0,7); if(m!==lastM){ if(lastM && i<n-8) g+=`<text class="ax" x="${x(i)}" y="${H-5}" text-anchor="middle">${new Date(d+'T12:00').toLocaleString(undefined,{month:'short'})}</text>`; lastM=m;}});
  const path=a=>a.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+','+y(v).toFixed(1)).join('');
  g+=`<path d="${path(gc.spy)}" fill="none" stroke="var(--ink3)" stroke-width="2"/>`;
  const colors = shown.map((_,i)=>`var(--c${i+1})`);
  shown.forEach((grp,i)=>{ g+=`<path d="${path(grp.series)}" fill="none" stroke="${colors[i]}" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${x(n-1)}" cy="${y(grp.series[n-1])}" r="4" fill="${colors[i]}" stroke="var(--surface)" stroke-width="2"/>`; });
  // End labels, only where they don't collide (the legend and tooltip carry the rest).
  const ends = [...shown.map((grp,i)=>({t:grp.name, v:grp.series[n-1], c:colors[i]})), {t:'SPY', v:gc.spy[n-1], c:'var(--ink3)'}].sort((a,b)=>b.v-a.v);
  let lastY=-99; for(const e of ends){ const yy=y(e.v); if(yy-lastY>=13){ g+=`<text class="lbl" x="${W-P.r+36}" y="${yy+4}">${esc(e.t.length>18?e.t.slice(0,17)+'\u2026':e.t)} ${Math.round(e.v-100)>=0?'+':''}${Math.round(e.v-100)}%</text>`; lastY=yy; } }
  g+=`<line id="gc-x" y1="${P.t}" y2="${H-P.b}" stroke="var(--ink3)" visibility="hidden"/><rect id="gc-hit" x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}" fill="transparent"/>`;
  const legend = `<div class="legend">${shown.map((grp,i)=>`<span><i style="background:${colors[i]}"></i>${esc(grp.name)}${grp.rank?` <span class="dim">#${grp.rank}</span>`:''}</span>`).join('')}<span><i style="background:var(--ink3)"></i>SPY</span></div>`;
  el.innerHTML = head + (shown.length ? legend + `<div class="chartwrap"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Industry group indexes over the last year against SPY">${g}</svg><div class="tip" id="gc-tip"></div></div>`
    : '<p class="note">None of your holdings is in a ranked group.</p>');
  el.querySelectorAll('button[data-g]').forEach(b=>b.onclick=()=>{groupView=b.dataset.g; groupChart();});
  if(!shown.length) return;
  const svg=el.querySelector('svg'), xl=el.querySelector('#gc-x'), tip=el.querySelector('#gc-tip');
  el.querySelector('#gc-hit').onmousemove = ev => {
    const pt=svg.createSVGPoint(); pt.x=ev.clientX; pt.y=ev.clientY; const loc=pt.matrixTransform(svg.getScreenCTM().inverse());
    const i=Math.max(0,Math.min(n-1,Math.round((loc.x-P.l)/((W-P.l-P.r)/(n-1)))));
    xl.setAttribute('x1',x(i)); xl.setAttribute('x2',x(i)); xl.setAttribute('visibility','visible');
    const vals=[...shown.map((grp,k)=>({t:grp.name,v:grp.series[i],c:colors[k]})),{t:'SPY',v:gc.spy[i],c:'var(--ink3)'}].sort((a,b)=>b.v-a.v);
    tip.innerHTML=`<b>${gc.dates[i]}</b><br>`+vals.map(o=>`<span class="key" style="border-color:${o.c}"></span>${esc(o.t)} <b>${(o.v-100>=0?'+':'')+(o.v-100).toFixed(1)}%</b>`).join('<br>');
    tip.style.display='block'; const box=svg.getBoundingClientRect(), px=ev.clientX-box.left;
    tip.style.left=(px>box.width*0.55?px-tip.offsetWidth-14:px+14)+'px'; tip.style.top='10px';
  };
  el.querySelector('#gc-hit').onmouseleave=()=>{xl.setAttribute('visibility','hidden'); tip.style.display='none';};
}

/* signals: this app's data, compact */
function sigs(r){
  const o=r.ours||{}, t=[];
  if(r.held) t.push(['H','You hold it']);
  if(o.pick) t.push(['P',`Discover pick ${o.pick.d}`]);
  if(o.book!==undefined && o.book>10) t.push(['B',`Contracted book ${sgn(o.book,0)} a year`]);
  if(o.insiders) t.push(['I',`${o.insiders.n} insiders buying`]);
  if(o.funds_net>0) t.push(['F','Hedge funds adding']);
  if(o.standout) t.push(['S',`Earnings standout ${o.standout.d}`]);
  return t.map(([k,title])=>`<span class="sig" title="${esc(title)}">${k}</span>`).join('');
}

/* lists */
const inZone = r => (r.st==='buy zone'||r.st==='breakout'||(r.st==='below pivot'&&r.vp>=-0.05)) && (r.c||0)>=70;
const VIEWS = [
  ['zone','Buy zone', L.rows.filter(inZone)],
  ['leaders','Leaders', L.rows.slice(0, L.leaders)],
  ['picks','Our picks', L.rows.filter(r=>r.ours.pick||r.ours.standout||r.ours.insiders)],
  ['held','Holdings', L.rows.filter(r=>r.held)],
  ['groups','Groups', L.groups],
];
let view = 'zone', selected = null, shown = [], group = null;
const HEAD = `<tr><th>Symbol</th><th class="num">Comp</th><th class="num">EPS</th><th class="num">RS</th>
  <th class="num">A/D</th><th>Signals</th></tr>`;
const GHEAD = `<tr><th class="num">Rank</th><th>Group</th><th class="num">Stocks</th><th>Composite 90+</th></tr>`;
const KEYS = [r=>r.t, r=>r.c, r=>r.eps, r=>r.rs, r=>r.ad===null?null:'EDCBA'.indexOf(r.ad), r=>sigs(r).length||null];
const GKEYS = [g=>g.rank, g=>g.name, g=>g.n, g=>g.leaders.length||null];

function seg(){
  $('#ms-seg').innerHTML = VIEWS.map(([k,label,rows])=>
    `<button role="tab" aria-selected="${k===view}" class="${k===view?'on':''}" data-v="${k}">${label}<span class="n">${rows.length}</span></button>`).join('');
  if(group) $('#ms-seg').innerHTML += `<button class="on" id="ms-clear" title="Show all leaders">${esc(group)} ×</button>`;
  document.querySelectorAll('#ms-seg button[data-v]').forEach(b=>b.onclick=()=>{view=b.dataset.v; group=null; $('#ms-q').value=''; list();});
  if(group) $('#ms-clear').onclick=()=>{group=null; list();};
}
function list(){
  seg();
  const q = $('#ms-q').value.trim().toUpperCase();
  const all = VIEWS.find(v=>v[0]===view)[2];
  if(view==='groups'){
    $('#ms-head').innerHTML = GHEAD;
    const rows = all.filter(g=>!q||g.name.toUpperCase().includes(q));
    sortable('#ms-rows', rows, rs=>{ $('#ms-rows').innerHTML = rs.map(g=>`<tr data-g="${esc(g.name)}">
      <td class="num">${g.rank}</td><td>${esc(g.name)}</td><td class="num">${g.n}</td>
      <td class="dim">${g.leaders.slice(0,5).map(esc).join(', ')||'—'}</td></tr>`).join('');
      document.querySelectorAll('#ms-rows tr').forEach(tr=>tr.onclick=()=>{
        view='leaders'; group=tr.dataset.g; $('#ms-q').value=''; list();});
    }, GKEYS, [2,3]);
    return;
  }
  $('#ms-head').innerHTML = HEAD;
  const base = view==='leaders' && group ? L.rows.filter(r=>r.ind===group) : all;
  shown = base.filter(r=>!q||r.t.includes(q)||(r.ind||'').toUpperCase().includes(q));
  sortable('#ms-rows', shown, rs=>{
    shown = rs;
    $('#ms-rows').innerHTML = rs.map(r=>`<tr data-t="${r.t}" class="${r.t===selected?'on':''}">
      <td><b>${esc(r.t)}</b>${r.rsh?' <span class="dim" title="RS line at a 52-week high">↗</span>':''}</td>
      <td class="num">${rate(r.c)}</td><td class="num">${rate(r.eps)}</td><td class="num">${rate(r.rs)}</td>
      <td class="num">${r.ad?esc(r.ad):dash}</td><td>${sigs(r)}</td></tr>`).join('') ||
      '<tr><td colspan="6" class="dim">Nothing in this list today.</td></tr>';
    document.querySelectorAll('#ms-rows tr[data-t]').forEach(tr=>tr.onclick=()=>pick(tr.dataset.t));
  }, KEYS, [1,2,3,4,5]);
  if(shown.length && !shown.some(r=>r.t===selected)) pick(shown[0].t, false);
}
$('#ms-q').oninput = list;
document.addEventListener('keydown', e=>{
  if(!root.classList.contains('on') || e.target.tagName==='INPUT' || !shown.length) return;
  if(e.key!=='ArrowDown' && e.key!=='ArrowUp') return;
  const i = shown.findIndex(r=>r.t===selected);
  const j = Math.max(0, Math.min(shown.length-1, i + (e.key==='ArrowDown'?1:-1)));
  e.preventDefault(); pick(shown[j].t);
});

/* detail */
const BY = Object.fromEntries(L.rows.map(r=>[r.t,r]));
function badge(k, v, max, extra='', fill=null){
  const f = fill ?? (typeof v==='number' ? v : null);
  const m = f===null ? '' : `<div class="meter"><i style="width:${Math.max(3,f/max*100)}%"></i></div>`;
  return `<div class="badge"><div class="k">${k}</div><div class="v">${v??'—'}${extra}</div>${m}</div>`;
}
function pick(t, scroll=true){
  selected = t;
  document.querySelectorAll('#ms-rows tr').forEach(tr=>tr.classList.toggle('on', tr.dataset.t===t));
  const tr = document.querySelector(`#ms-rows tr[data-t="${t}"]`);
  if(tr && scroll) tr.scrollIntoView({block:'nearest'});
  const r = BY[t], o = r.ours||{}, h = HOLD[t];
  const status = r.st ? `<span class="pill" style="color:var(--${r.st==='buy zone'||r.st==='breakout'?'good':r.st==='extended'?'warn':'ink2'});border-color:currentColor">${esc(r.st)}</span>` : '';
  let d = `<div class="dhead"><h2>${esc(t)}</h2><span class="px">$${fmt(r.p)}</span>
    <span class="dim">${esc(r.ind||'Industry unknown')}${r.g?` · group #${r.g} of ${L.groups_ranked}`:''}</span>${status}
    ${r.held?'<span class="pill">held</span>':''}</div>`;
  const gpct = r.g ? Math.round(99 - (r.g-1)/Math.max(1,L.groups_ranked-1)*98) : null;
  d += '<div class="badges">' + badge('Composite', r.c, 99) + badge('EPS', r.eps, 99) + badge('RS', r.rs, 99) +
    badge('Group rank', r.g?'#'+r.g:null, 99, '', gpct) +
    `<div class="badge"><div class="k">Acc/Dis</div><div class="v">${r.ad??'—'}</div></div>` +
    badge('Screen', o.score??null, 100, o.score!==undefined?'<small>/100</small>':'') + '</div>';
  d += `<div class="legend">
    <span><i class="key bar" style="background:var(--s1)"></i>Up day</span>
    <span><i class="key bar" style="background:var(--s2)"></i>Down day</span>
    <span><i class="key" style="border-color:var(--s3)"></i>50-day</span>
    <span><i class="key" style="border-color:var(--ink3)"></i>200-day</span>
    ${r.piv?'<span><i class="key" style="border-color:var(--ink2)"></i>Pivot · buy zone</span>':''}</div>
    <div class="chartwrap"><div id="ms-chart"></div><div class="tip" id="ms-tip"></div></div>`;
  const facts = [];
  if(r.base) facts.push(['Base', `${esc(r.base)}, ${r.wk} weeks, ${(r.dep*100).toFixed(0)}% deep`]);
  if(r.piv) facts.push(['Buy point', `$${fmt(r.piv)} · zone to $${fmt(r.piv*1.05)} · price ${sgn(r.vp*100)} vs pivot`]);
  facts.push(['Off 52-week high', r.off===null?'—':sgn(r.off*100)]);
  if(r.q1!==null&&r.q1!==undefined) facts.push(['Latest quarter EPS', `${sgn(r.q1*100,0)} year over year`]);
  if(h) facts.push(['Your holding', `${h.verdict?esc(h.verdict)+(h.conf?` ${h.conf}/10`:''):'no review'} · ${h.value!==null?money(h.value):'—'}${h.pl!==null?` · P/L ${sgn(h.pl)}`:''}`]);
  if(o.pick) facts.push(['Discover pick', `#${o.pick.rank} on ${o.pick.d}${o.pick.conv?` · conviction ${o.pick.conv}/10`:''}${o.pick.ev!==null&&o.pick.ev!==undefined?` · EV ${sgn(o.pick.ev)}`:''}`]);
  if(o.book!==undefined) facts.push(['Contracted book', `${sgn(o.book,0)} year over year (SEC)`]);
  if(o.rev) facts.push(['EPS revisions, 30 days', esc(o.rev)]);
  if(o.insiders) facts.push(['Insider buying', `${o.insiders.n} insiders, ${money(o.insiders.usd)} since ${esc(o.insiders.d)}`]);
  if(o.standout) facts.push(['Earnings standout', `reported ${esc(o.standout.d)}${o.standout.react!=null?` · ${sgn(o.standout.react)} vs SPY`:''}${o.standout.rev!=null?` · estimates ${sgn(o.standout.rev)}`:''}`]);
  if(o.funds) facts.push(['Hedge funds (13F)', esc(o.funds)]);
  d += '<div class="facts">' + facts.map(([k,v])=>`<div class="fact"><div class="k">${k}</div>${v}</div>`).join('') + '</div>';
  d += companyHtml(t) + newsHtml(t);
  if(D.views && D.views[t]) d += `<h3>Our long-term view</h3><div class="viewtext">${esc(D.views[t])}</div>`;
  d += '<h3>Rating history</h3><div class="chartwrap"><div id="ms-hist"></div><div class="tip" id="ms-htip"></div></div>';
  $('#ms-detail').innerHTML = d;
  chart(t, r);
  history(t);
  score();
  viewMap();
}

/* Composite and RS over time, from the daily history (one 1-99 scale) */
function history(t){
  const h = (L.history||{})[t], el = $('#ms-hist');
  if(!h){ el.innerHTML = '<p class="note">Builds up from the daily history: a few mornings until the first line appears.</p>'; return; }
  const W=Math.max(320, Math.round(el.clientWidth||760)), P={l:8,r:58,t:8,b:20}, H=150;
  const n=h.d.length, x=i=>P.l+(n===1?0:i*(W-P.l-P.r)/(n-1)), y=v=>P.t+(99-v)/98*(H-P.t-P.b);
  let g='';
  for(const v of [1,25,50,75,99]) g+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${y(v)}" y2="${y(v)}"/>
    <text class="ax" x="${W-P.r+6}" y="${y(v)+3.5}">${v}</text>`;
  g+=`<text class="ax" x="${P.l}" y="${H-5}">${h.d[0]}</text><text class="ax" x="${W-P.r}" y="${H-5}" text-anchor="end">${h.d[n-1]}</text>`;
  const line=(arr,c)=>{let p='',on=false; arr.forEach((v,i)=>{ if(v===null){on=false;return;} p+=(on?'L':'M')+x(i).toFixed(1)+','+y(v).toFixed(1); on=true;});
    return `<path d="${p}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;};
  g+=line(h.c,'var(--s1)')+line(h.rs,'var(--s2)');
  for(const [arr,c] of [[h.c,'var(--s1)'],[h.rs,'var(--s2)']]){const i=[...arr.keys()].reverse().find(k=>arr[k]!==null);
    if(i!==undefined) g+=`<circle cx="${x(i)}" cy="${y(arr[i])}" r="4" fill="${c}" stroke="var(--surface)" stroke-width="2"/>`;}
  g+=`<line id="ms-hx" y1="${P.t}" y2="${H-P.b}" stroke="var(--ink3)" visibility="hidden"/>
    <rect id="ms-hhit" x="${P.l}" y="${P.t}" width="${W-P.l-P.r}" height="${H-P.t-P.b}" fill="transparent"/>`;
  el.innerHTML = `<div class="legend"><span><i class="key" style="border-color:var(--s1)"></i>Composite</span>
    <span><i class="key" style="border-color:var(--s2)"></i>RS</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(t)} Composite and RS rating over time">${g}</svg>`;
  const svg=el.querySelector('svg'), hx=el.querySelector('#ms-hx'), tip=$('#ms-htip');
  el.querySelector('#ms-hhit').onmousemove = ev => {
    const pt=svg.createSVGPoint(); pt.x=ev.clientX; pt.y=ev.clientY; const loc=pt.matrixTransform(svg.getScreenCTM().inverse());
    const i=Math.max(0,Math.min(n-1,Math.round((loc.x-P.l)/((W-P.l-P.r)/Math.max(1,n-1)))));
    hx.setAttribute('x1',x(i)); hx.setAttribute('x2',x(i)); hx.setAttribute('visibility','visible');
    tip.innerHTML = `<b>${h.d[i]}</b><br>Composite <b>${h.c[i]??'—'}</b> · RS <b>${h.rs[i]??'—'}</b>`;
    tip.style.display='block'; const box=el.getBoundingClientRect(), px=ev.clientX-box.left;
    tip.style.left=(px>box.width*0.6?px-tip.offsetWidth-14:px+14)+'px'; tip.style.top='28px';
  };
  el.querySelector('#ms-hhit').onmouseleave = ()=>{hx.setAttribute('visibility','hidden'); tip.style.display='none';};
}

/* our view (fundamentals) against the market's (Composite), both ranked
   1-99 among the ~600 stocks the screen keeps fundamentals for */
function viewMap(){
  const el = $('#ms-map'), pts = L.view_map || [];
  if(!pts.length){ el.innerHTML = '<h2>Our view vs the market’s</h2><p class="note">Needs cached fundamentals, which the nightly job keeps.</p>'; return; }
  const W=Math.max(320, Math.round(el.clientWidth||760)-36), H=Math.min(460, Math.round(W*0.62)), P={l:34,r:12,t:12,b:34};
  const x=v=>P.l+(v-1)/98*(W-P.l-P.r), y=v=>P.t+(99-v)/98*(H-P.t-P.b);
  let g='';
  for(const v of [1,25,50,75,99]) g+=`<line class="gl" x1="${x(v)}" x2="${x(v)}" y1="${P.t}" y2="${H-P.b}"/>
    <line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${y(v)}" y2="${y(v)}"/>
    <text class="ax" x="${x(v)}" y="${H-P.b+14}" text-anchor="middle">${v}</text><text class="ax" x="${P.l-6}" y="${y(v)+3.5}" text-anchor="end">${v}</text>`;
  g+=`<line x1="${x(50)}" x2="${x(50)}" y1="${P.t}" y2="${H-P.b}" stroke="var(--ink3)"/><line x1="${P.l}" x2="${W-P.r}" y1="${y(50)}" y2="${y(50)}" stroke="var(--ink3)"/>`;
  const q=(tx,ty,anchor,text)=>`<text class="lbl" x="${tx}" y="${ty}" text-anchor="${anchor}" style="font-weight:600">${text}</text>`;
  g+=q(W-P.r-6,P.t+14,'end','Both strong')+q(P.l+6,P.t+14,'start','Market likes it, we don’t')+
     q(W-P.r-6,H-P.b-8,'end','We like it, market doesn’t')+q(P.l+6,H-P.b-8,'start','Both weak');
  g+=`<text class="ax" x="${(P.l+W-P.r)/2}" y="${H-4}" text-anchor="middle">Our view: fundamentals, book, revisions (rank) →</text>`;
  const order = [...pts].sort((a,b)=>(a.held||a.pick)-(b.held||b.pick));
  for(const p of order){
    const on = p.t===selected, key = p.held?'var(--s1)':p.pick?'var(--s2)':'var(--ink3)';
    const r = on?7:(p.held||p.pick)?5:4;
    g+=`<circle data-t="${p.t}" cx="${x(p.x)}" cy="${y(p.y)}" r="${r}" fill="${key}" opacity="${p.held||p.pick||on?1:.35}"
      stroke="var(--surface)" stroke-width="2" style="cursor:pointer"><title>${esc(p.t)}: our view ${p.x}, Composite ${p.comp} (rank ${p.y})${p.ind?' · '+esc(p.ind):''}</title></circle>`;
    if(p.held||on) g+=`<text class="lbl" x="${x(p.x)+8}" y="${y(p.y)+4}" style="pointer-events:none${on?';font-weight:600':''}">${esc(p.t)}</text>`;
  }
  el.innerHTML = `<h2>Our view vs the market’s</h2>
    <p class="note">Across: this app’s read of the business (the screen’s fundamentals score, contracted-book growth and EPS
    revisions). Up: the IBD-style Composite, which is mostly price strength. Both are ranked 1–99 among the ${pts.length}
    stocks the screen keeps fundamentals for. Disagreements are the interesting part: <b>we like it, market doesn’t</b>
    is either early or wrong, so check the thesis; <b>market likes it, we don’t</b> asks what we’re missing.</p>
    <div class="legend"><span><i style="background:var(--s1)"></i>Your holdings</span><span><i style="background:var(--s2)"></i>Our picks</span>
    <span><i style="background:var(--ink3)"></i>Other stocks</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Our fundamental view against the Composite rating">${g}</svg>`;
  el.querySelectorAll('circle[data-t]').forEach(c=>c.onclick=()=>{ if(BY[c.dataset.t]){ pick(c.dataset.t);
    $('#ms-detail').scrollIntoView({behavior:'smooth',block:'start'}); }});
}

/* how the logged buy-zone signals have done against SPY: the selected
   stock's by default, every stock's on the "All stocks" switch */
let scope = 'stock';
const SC = L.scorecard || {};
const hlabel = h => ({21:'1 month',63:'3 months',126:'6 months'})[h] || h+' sessions';
function summarize(sigs){
  return (L.horizons||[]).map(h=>{
    const done = sigs.map(x=>x.h[h]).filter(v=>v && v.done && v.ret!==null && v.spy!==null);
    const ex = done.map(v=>v.ret-v.spy);
    return {h, n:done.length, avg_excess: ex.length? ex.reduce((a,b)=>a+b,0)/ex.length : null,
      beat: ex.length? Math.round(ex.filter(e=>e>0).length/ex.length*100) : null};
  });
}
function score(){
  const el = $('#ms-score'), all = SC.signals || [];
  const mine = all.filter(x=>x.t===selected);
  const sigs = scope==='stock' ? mine : all;
  const summary = scope==='stock' ? summarize(mine) : (SC.summary || []);
  const who = scope==='stock' ? esc(selected||'this stock') : 'every stock';
  let out = `<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px">
    <h2 style="margin:0">How the buy-zone signals have done: ${who}</h2>
    <div class="seg" style="margin:0" role="tablist" aria-label="Signals scope">
     <button data-s="stock" class="${scope==='stock'?'on':''}" aria-selected="${scope==='stock'}">${esc(selected||'This stock')}<span class="n">${mine.length}</span></button>
     <button data-s="all" class="${scope==='all'?'on':''}" aria-selected="${scope==='all'}">All stocks<span class="n">${SC.total||0}</span></button></div></div>
    <p class="note" style="margin-top:8px">Each time a stock with Composite 80+ enters a buy zone or breaks out, it is logged
    (once per 30 days) and graded against SPY over the same sessions.${SC.backfilled?` ${SC.backfilled} of the
    ${SC.total} were rebuilt from past prices and SEC filings (marked <span class="sig">R</span>) using today’s list of
    stocks, which leaves out companies that have since shrunk or delisted, so read them as indicative; the rest were
    recorded live.`:''}</p>`;
  if(!all.length) out += '<p class="note">Nothing logged yet: signals appear from the morning runs.</p>';
  else if(!sigs.length) out += `<p class="note">No signal logged for ${who} yet. It will appear here the first time it
    enters a buy zone or breaks out with Composite 80+. Switch to <b>All stocks</b> for the ${SC.total} logged so far.</p>`;
  if(sigs.length){
    out += '<div class="tiles">' + summary.map(x=>`<div class="tile"><div class="k">${hlabel(x.h)} later</div>
      <div class="v">${x.avg_excess===null?'\u2014':(x.avg_excess>=0?'+':'')+x.avg_excess.toFixed(1)+' pts'}</div>
      <div class="note" style="margin:4px 0 0">${x.n?`vs SPY, average of ${x.n} · ${x.beat}% beat SPY`:'no signal is this old yet'}</div></div>`).join('') + '</div>';
    out += `<div class="table-scroll"><table><thead><tr><th>Date</th><th>Ticker</th><th>Signal</th><th class="num">Entry</th>
      <th class="num">Comp</th><th class="num">RS</th>${(L.horizons||[]).map(h=>`<th class="num">${hlabel(h)}</th>`).join('')}</tr></thead>
      <tbody id="ms-sig"></tbody></table></div>`;
  }
  el.innerHTML = out;
  el.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>{scope=b.dataset.s; score();});
  if(!sigs.length) return;
  const cell = x => !x || x.ret===null ? dash : `${num(x.ret)}${x.spy===null?'':` <span class="dim">SPY ${(x.spy>=0?'+':'')+x.spy.toFixed(1)}%</span>`}${x.done?'':' <span class="dim">so far</span>'}`;
  sortable('#ms-sig', sigs, rs=>{
    $('#ms-sig').innerHTML = rs.map(x=>`<tr data-t="${x.t}" class="${x.t===selected?'on':''}"><td class="dim">${x.d}</td>
      <td><b>${esc(x.t)}</b>${x.bf?' <span class="sig" title="Rebuilt from past prices, not recorded live">R</span>':''}</td>
      <td>${esc(x.st)}</td><td class="num">$${fmt(x.p)}</td><td class="num">${x.c??dash}</td>
      <td class="num">${x.rs??dash}</td>${(L.horizons||[]).map(h=>`<td class="num">${cell(x.h[h])}</td>`).join('')}</tr>`).join('');
    // A row opens that stock above, when it is on the page.
    document.querySelectorAll('#ms-sig tr').forEach(tr=>tr.onclick=()=>{ if(BY[tr.dataset.t]){ pick(tr.dataset.t);
      $('#ms-detail').scrollIntoView({behavior:'smooth',block:'start'}); }});
  }, [x=>x.d, x=>x.t, x=>x.st, x=>x.p, x=>x.c, x=>x.rs,
      ...(L.horizons||[]).map(h=>x=>!x.h[h]||x.h[h].ret===null||x.h[h].spy===null?null:x.h[h].ret-x.h[h].spy)],
    [0,3,4,5,6,7,8]);
}

/* company profile (fundamentals cache) and this morning's top news */
const big = v => v>=1e12?'$'+(v/1e12).toFixed(2)+'T':v>=1e9?'$'+(v/1e9).toFixed(1)+'B':'$'+(v/1e6).toFixed(0)+'M';
const pctv = v => v===null||v===undefined ? '\u2014' : (v>=0?'+':'')+(v*100).toFixed(0)+'%';
function companyHtml(t){
  const c = (L.profiles||{})[t];
  if(!c) return '<h3>Company</h3><p class="note">No profile cached yet: it fills in with the nightly fundamentals refresh.</p>';
  const facts = [];
  if(c.market_cap) facts.push(['Market cap', big(c.market_cap)]);
  if(c.forward_pe || c.trailing_pe) facts.push(['P/E', `${c.forward_pe?c.forward_pe.toFixed(1)+' forward':''}${c.forward_pe&&c.trailing_pe?' · ':''}${c.trailing_pe?c.trailing_pe.toFixed(1)+' trailing':''}`]);
  if(c.revenue_growth_yoy!==undefined) facts.push(['Growth, year over year', `revenue ${pctv(c.revenue_growth_yoy)}${c.earnings_growth_yoy!==undefined?` · earnings ${pctv(c.earnings_growth_yoy)}`:''}`]);
  if(c.operating_margin!==undefined) facts.push(['Margins', `operating ${pctv(c.operating_margin)}${c.profit_margin!==undefined?` · net ${pctv(c.profit_margin)}`:''}`]);
  // Upside from today's price: the cached figure was taken at an older one.
  const px = (BY[t]||{}).p, up = c.analyst_target_mean && px ? (c.analyst_target_mean/px-1)*100 : null;
  if(c.analyst_target_mean) facts.push(['Analysts', `${c.analyst_count||'?'} covering${c.analyst_recommendation?' · '+esc(String(c.analyst_recommendation).replace('_',' ')):''} · target $${fmt(c.analyst_target_mean)}${up!==null?` (${sgn(up)} from here)`:''}`]);
  // A past date means the company has reported and not yet set the next one.
  if(c.next_earnings) facts.push(['Next earnings', c.next_earnings >= L.as_of ? esc(c.next_earnings) : `not announced yet <span class="dim">(last ${esc(c.next_earnings)})</span>`]);
  if(c.hq || c.employees) facts.push(['Headquarters', `${esc(c.hq||'\u2014')}${c.employees?` · ${Number(c.employees).toLocaleString()} employees`:''}`]);
  let site = '';
  try{ if(c.website){ const u=new URL(c.website); if(/^https?:$/.test(u.protocol)) site=` · <a href="${esc(u.href)}" target="_blank" rel="noopener noreferrer">${esc(u.hostname.replace(/^www\./,''))}</a>`; } }catch(e){}
  return `<h3>Company${c.name?`: ${esc(c.name)}`:''}${site}</h3>` +
    (c.summary ? `<p style="font-size:13.5px;color:var(--ink2);margin:0 0 8px">${esc(c.summary)}${c.summary.length>=690?'\u2026':''}</p>` : '') +
    '<div class="facts">' + facts.map(([k,v])=>`<div class="fact"><div class="k">${k}</div>${v}</div>`).join('') + '</div>';
}
function newsHtml(t){
  const n = (L.news||{})[t];
  if(!n) return '<h3>Top news</h3><p class="note">News is gathered each morning for holdings, picks, the top leaders and stocks near a buy point; this one is not among them today.</p>';
  if(!n.items.length) return `<h3>Top news</h3><p class="note">Nothing about the company itself in the last two weeks (checked ${esc(n.fetched)}). Syndicated market commentary that doesn’t name it is left out.</p>`;
  const link = (u, text) => { try{ const x=new URL(u); if(/^https?:$/.test(x.protocol)) return `<a href="${esc(x.href)}" target="_blank" rel="noopener noreferrer">${text}</a>`; }catch(e){} return text; };
  return `<h3>Top news <span class="dim" style="font-weight:400">· most material first, checked ${esc(n.fetched)}</span></h3><ol style="padding-left:20px;margin:0">` +
    n.items.map(i=>`<li style="margin:0 0 8px"><b>${link(i.url, esc(i.title))}</b>
      <div class="dim" style="font-size:12px">${esc(i.source||'')}${i.d?' · '+esc(i.d):''}</div>
      ${i.snippet?`<div style="font-size:12.5px;color:var(--ink2)">${esc(i.snippet)}</div>`:''}</li>`).join('') + '</ol>';
}

/* chart: price / RS line / volume panes on one date axis */
function niceTicks(lo, hi, n){
  const span = hi-lo, step0 = span/n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=step0) || mag*10;
  const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(6)); return out;
}
function chart(t, r){
  const s = (L.charts||{})[t], cal = L.calendar||[];
  const el = $('#ms-chart');
  if(!s){ el.innerHTML = '<p class="note">No stored price history for this stock yet.</p>'; return; }
  // Drawn at the container's own width, so axis text stays 10.5px on a phone.
  const W=Math.max(320, Math.round(el.clientWidth||760)), P={l:8,r:58}, gap=12;
  const H1=W<500?200:250, H2=64, H3=78, top=6, axis=20;
  const y1=top, y2=y1+H1+gap, y3=y2+H2+gap, H=y3+H3+axis;
  const n=cal.length, cw=(W-P.l-P.r)/n, x=i=>P.l+(i+0.5)*cw;
  const fin = a => a.filter(v=>v!==null&&v!==undefined);
  let lo=Math.min(...fin(s.l)), hi=Math.max(...fin(s.h));
  for(const v of [...fin(s.m50),...fin(s.m200)]) if(v>lo*0.85&&v<hi*1.15){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  if(r.piv && r.piv<hi*1.12 && r.piv>lo*0.9){hi=Math.max(hi,r.piv*1.05);lo=Math.min(lo,r.piv);}
  const pad=(hi-lo)*0.04; lo-=pad; hi+=pad;
  const py=v=>y1+(hi-v)/(hi-lo)*H1;
  const rsv=fin(s.rs), rlo=Math.min(...rsv), rhi=Math.max(...rsv), rpad=(rhi-rlo)*0.08||1;
  const ry=v=>y2+(rhi+rpad-v)/((rhi-rlo)+2*rpad)*H2;
  const vmax=Math.max(1,...fin(s.v)), vy=v=>y3+H3-(v/vmax)*H3;
  let g='';
  for(const v of niceTicks(lo,hi,5)) g+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${py(v)}" y2="${py(v)}"/>
    <text class="ax" x="${W-P.r+6}" y="${py(v)+3.5}">${v>=1000?v.toLocaleString():v}</text>`;
  g+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${y2+H2}" y2="${y2+H2}"/><text class="ax" x="${W-P.r+6}" y="${y2+10}">RS line</text>`;
  g+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${y3+H3}" y2="${y3+H3}"/>`;
  for(const v of niceTicks(0,vmax,2).filter(v=>v>0)) g+=`<text class="ax" x="${W-P.r+6}" y="${vy(v)+3.5}">${v>=1000?(v/1000).toFixed(v%1000?1:0)+'M':v+'K'}</text>
    <line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${vy(v)}" y2="${vy(v)}"/>`;
  let lastM='';
  cal.forEach((d,i)=>{const m=d.slice(0,7); if(m!==lastM){ if(lastM) g+=`<text class="ax" x="${x(i)}" y="${H-5}" text-anchor="middle">${
    new Date(d+'T12:00').toLocaleString(undefined,{month:'short'})}</text>`; lastM=m;}});
  if(r.piv){
    const a=py(r.piv*1.05), b=py(r.piv);
    g+=`<rect x="${P.l}" width="${W-P.l-P.r}" y="${a}" height="${Math.max(1,b-a)}" fill="var(--s3)" opacity=".1"/>
      <line x1="${P.l}" x2="${W-P.r}" y1="${b}" y2="${b}" stroke="var(--ink2)" stroke-width="1"/>
      <text class="lbl" x="${P.l+4}" y="${b-4}">Pivot $${fmt(r.piv)}</text>`;
  }
  const up=i=> i>0 && s.c[i]!==null && s.c[i-1]!==null ? s.c[i]>=s.c[i-1] : true;
  const bw=Math.max(1,Math.min(cw*0.55,3));
  for(let i=0;i<n;i++){
    if(s.h[i]===null||s.l[i]===null||s.c[i]===null) continue;
    const col=up(i)?'var(--s1)':'var(--s2)';
    g+=`<line x1="${x(i)}" x2="${x(i)}" y1="${py(s.h[i])}" y2="${py(s.l[i])}" stroke="${col}" stroke-width="${Math.min(1.5,bw)}"/>
      <line x1="${x(i)}" x2="${x(i)+Math.max(2,cw*0.45)}" y1="${py(s.c[i])}" y2="${py(s.c[i])}" stroke="${col}" stroke-width="1.5"/>`;
    if(s.v[i]!==null){const top_=vy(s.v[i]); g+=`<rect x="${x(i)-bw/2}" y="${top_}" width="${bw}" height="${Math.max(0.5,y3+H3-top_)}" fill="${col}" rx="${Math.min(1,bw/2)}"/>`;}
  }
  const path=(arr,yf)=>{let p='',on=false; arr.forEach((v,i)=>{ if(v===null||v===undefined){on=false;return;}
    p+=(on?'L':'M')+x(i).toFixed(1)+','+yf(v).toFixed(1); on=true;}); return p;};
  g+=`<path d="${path(s.m200,py)}" fill="none" stroke="var(--ink3)" stroke-width="2" stroke-linejoin="round"/>`;
  g+=`<path d="${path(s.m50,py)}" fill="none" stroke="var(--s3)" stroke-width="2" stroke-linejoin="round"/>`;
  g+=`<path d="${path(s.rs,ry)}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>`;
  g+=`<path d="${path(s.va,vy)}" fill="none" stroke="var(--ink3)" stroke-width="1.5"/>`;
  const li=[...s.rs.keys()].reverse().find(i=>s.rs[i]!==null);
  if(li!==undefined) g+=`<circle cx="${x(li)}" cy="${ry(s.rs[li])}" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"><title>RS line ${r.rsh?'at a 52-week high':''}</title></circle>`;
  g+=`<line id="ms-x" x1="0" x2="0" y1="${y1}" y2="${y3+H3}" stroke="var(--ink3)" stroke-width="1" visibility="hidden"/>
    <rect id="ms-hit" x="${P.l}" y="${y1}" width="${W-P.l-P.r}" height="${y3+H3-y1}" fill="transparent"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(t)} daily price, RS line and volume, six months">${g}</svg>`;
  const svg=el.querySelector('svg'), hit=el.querySelector('#ms-hit'), xl=el.querySelector('#ms-x'), tip=$('#ms-tip');
  const move = ev => {
    const pt=svg.createSVGPoint(); pt.x=ev.clientX; pt.y=ev.clientY;
    const loc=pt.matrixTransform(svg.getScreenCTM().inverse());
    const i=Math.max(0,Math.min(n-1,Math.floor((loc.x-P.l)/cw)));
    xl.setAttribute('x1',x(i)); xl.setAttribute('x2',x(i)); xl.setAttribute('visibility','visible');
    const chg = i>0&&s.c[i]!==null&&s.c[i-1]? (s.c[i]/s.c[i-1]-1)*100 : null;
    tip.innerHTML = `<b>${cal[i]}</b><br>Close <b>$${fmt(s.c[i])}</b> ${chg===null?'':`<span class="${chg>=0?'pos':'neg'}">${sgn(chg,2)}</span>`}<br>
      High $${fmt(s.h[i])} · Low $${fmt(s.l[i])}<br>50-day $${fmt(s.m50[i])} · 200-day $${fmt(s.m200[i])}<br>
      Volume ${s.v[i]===null?'—':(s.v[i]*1000).toLocaleString()} <span class="dim">(avg ${s.va[i]===null?'—':(s.va[i]*1000).toLocaleString()})</span><br>
      RS line ${fmt(s.rs[i],1)}`;
    tip.style.display='block';
    const box=el.getBoundingClientRect(), px=ev.clientX-box.left;
    tip.style.left = (px > box.width*0.6 ? px - tip.offsetWidth - 14 : px + 14) + 'px';
    tip.style.top = '8px';
  };
  hit.onmousemove = move;
  hit.onmouseleave = ()=>{ xl.setAttribute('visibility','hidden'); tip.style.display='none'; };
}

list();
groupChart();
})();
"""
