// Shell C — ESTAÇÃO
// Research instrument. Three-pane IDE/terminal layout, mono-led with
// occasional serif for film titles. Dense rows, hotkeys visible,
// hybrid-retrieval bars on the inspector. Phosphor cyan accent.

const SC_PALETTE = {
  ink:       '#0D0E10',
  pane:      '#0F1114',
  surface:   '#13161A',
  raised:    '#1A1D22',
  hairline:  '#23262C',
  hairline2: '#343941',
  frost:     '#E8E9EC',
  frostDim:  '#B8BAC0',
  muted:     '#6B7280',
  faint:     '#4B5058',
  accent:    '#7BB3D9',
  accentDim: '#4A6E87',
  good:      '#6FBE94',
  warn:      '#D9A85C',
  err:       '#C26B6B',
};

const SC_CSS = `
.shell-C { all: initial; box-sizing: border-box; }
.shell-C *, .shell-C *::before, .shell-C *::after { box-sizing: border-box; }
.shell-C * { border-radius: 0 !important; }
.shell-C {
  --ink: ${SC_PALETTE.ink};
  --pane: ${SC_PALETTE.pane};
  --surface: ${SC_PALETTE.surface};
  --raised: ${SC_PALETTE.raised};
  --line: ${SC_PALETTE.hairline};
  --line2: ${SC_PALETTE.hairline2};
  --frost: ${SC_PALETTE.frost};
  --frost-dim: ${SC_PALETTE.frostDim};
  --muted: ${SC_PALETTE.muted};
  --faint: ${SC_PALETTE.faint};
  --accent: ${SC_PALETTE.accent};
  --accent-dim: ${SC_PALETTE.accentDim};
  --good: ${SC_PALETTE.good};
  --warn: ${SC_PALETTE.warn};
  --err: ${SC_PALETTE.err};
  --mono: 'JetBrains Mono', 'Berkeley Mono', 'Courier New', monospace;
  --serif: 'Source Serif 4', 'Newsreader', Georgia, serif;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  display: grid;
  grid-template-rows: 28px 32px 1fr 26px;
  height: 100%; width: 100%;
  background: var(--ink); color: var(--frost-dim);
  font-family: var(--mono); font-size: 12px; line-height: 1.45;
  font-variant-numeric: tabular-nums; font-feature-settings: 'calt' 0;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* ─── TOP STATUS ────────────────────────────────────────────────────── */
.shell-C .topstatus {
  display:flex; align-items:center; justify-content: space-between;
  padding: 0 14px; border-bottom: 1px solid var(--line);
  background: var(--pane);
  font-size: 10.5px; color: var(--muted); letter-spacing: 0.02em;
}
.shell-C .topstatus .left { display:flex; align-items:center; gap: 14px; }
.shell-C .topstatus .right { display:flex; align-items:center; gap: 18px; }
.shell-C .topstatus .brand {
  color: var(--frost); display:flex; align-items:center; gap: 8px;
}
.shell-C .topstatus .brand .glyph { color: var(--accent); }
.shell-C .topstatus .crumb { color: var(--muted); }
.shell-C .topstatus .crumb .sep { color: var(--faint); margin: 0 4px; }
.shell-C .topstatus .crumb .cur { color: var(--frost); }
.shell-C .topstatus .dot { display:inline-block; width:6px; height:6px; background: var(--good); margin-right:5px; vertical-align: 1px; }
.shell-C .topstatus .dot.warn { background: var(--warn); }

/* ─── TAB BAR (modeline) ────────────────────────────────────────────── */
.shell-C .tabline {
  display:flex; align-items:center; gap: 0;
  padding: 0 14px; border-bottom: 1px solid var(--line);
  background: var(--surface);
}
.shell-C .tab {
  display:flex; align-items:center; gap: 6px;
  padding: 0 14px; height: 100%;
  font-size: 11.5px; color: var(--muted); cursor: pointer;
  border-right: 1px solid var(--line);
  letter-spacing: 0.01em;
}
.shell-C .tab .k {
  color: var(--faint); font-size: 10px;
}
.shell-C .tab.active {
  color: var(--frost); background: var(--ink);
  box-shadow: inset 0 2px 0 var(--accent);
}
.shell-C .tab.active .k { color: var(--accent); }
.shell-C .tab .pip {
  background: var(--accent-dim); color: var(--ink);
  padding: 0 5px; font-size: 10px; margin-left: 4px;
}
.shell-C .tab .pip.warn { background: var(--warn); }
.shell-C .tabline .grow { flex: 1; }
.shell-C .tabline .modes {
  display:flex; align-items:center; gap: 18px;
  font-size: 10.5px; color: var(--muted); padding-right: 4px;
}
.shell-C .tabline .modes b { color: var(--accent); font-weight: 400; }

/* ─── BODY: 3 PANES ─────────────────────────────────────────────────── */
.shell-C .body {
  display: grid; grid-template-columns: 280px 1fr 340px;
  overflow: hidden;
}

/* LEFT PANE — film tree */
.shell-C .lp { border-right: 1px solid var(--line); display:flex; flex-direction:column; overflow:hidden; }
.shell-C .lp .head {
  display:flex; align-items:center; justify-content:space-between;
  padding: 8px 12px; border-bottom: 1px solid var(--line);
  font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); background: var(--pane);
}
.shell-C .lp .head .ct { color: var(--frost); }
.shell-C .lp .tree { padding: 6px 0; overflow-y: auto; flex:1; }
.shell-C .tree-row {
  display:grid; grid-template-columns: 16px 1fr auto auto;
  align-items:center; gap: 6px;
  padding: 3px 12px; font-size: 12px;
  color: var(--frost-dim); cursor: pointer;
}
.shell-C .tree-row:hover { background: var(--surface); }
.shell-C .tree-row.active { background: var(--surface); color: var(--frost); }
.shell-C .tree-row.active::before {
  content: ''; position: absolute; left: 0; width: 2px; height: 1.5em;
  background: var(--accent);
}
.shell-C .tree-row { position: relative; }
.shell-C .tree-row .arrow { color: var(--faint); font-size: 10px; text-align: center; }
.shell-C .tree-row.open .arrow { color: var(--accent); }
.shell-C .tree-row .year { color: var(--muted); font-size: 10.5px; }
.shell-C .tree-row .ct { color: var(--faint); font-size: 10.5px; }
.shell-C .tree-row.indent { padding-left: 30px; }
.shell-C .tree-row.indent .arrow { color: var(--good); }
.shell-C .tree-row.indent .ct { color: var(--good); }
.shell-C .tree-row.proc { color: var(--warn); }
.shell-C .tree-row.proc .arrow,
.shell-C .tree-row.proc .ct { color: var(--warn); }
.shell-C .lp .footer {
  padding: 8px 12px; border-top: 1px solid var(--line);
  font-size: 10px; color: var(--muted);
  display:grid; grid-template-columns: 1fr auto; row-gap: 3px;
}
.shell-C .lp .footer .v { color: var(--frost); justify-self: end; }

/* CENTER PANE — search + results */
.shell-C .cp { display:flex; flex-direction:column; overflow:hidden; }
.shell-C .cmdline {
  padding: 14px 22px 10px; border-bottom: 1px solid var(--line);
  background: var(--pane);
}
.shell-C .prompt {
  display: flex; align-items:center; gap: 12px;
  border-bottom: 1px solid var(--line2); padding-bottom: 10px;
}
.shell-C .prompt .pre {
  color: var(--accent); font-size: 13px;
}
.shell-C .prompt .q {
  flex: 1; background: transparent; border: none; outline: none;
  color: var(--frost); font-family: var(--mono); font-size: 14.5px;
  letter-spacing: -0.005em; caret-color: var(--accent);
}
.shell-C .prompt .q::placeholder { color: var(--faint); }
.shell-C .prompt .submit {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--ink); background: var(--accent);
  border: none; padding: 5px 9px; cursor: pointer;
}
.shell-C .knobs {
  display:flex; align-items:center; gap: 14px; padding-top: 8px;
  font-size: 11px; color: var(--muted); flex-wrap: wrap;
}
.shell-C .knobs .k { color: var(--faint); }
.shell-C .knobs .v { color: var(--frost); }
.shell-C .knobs .seg { display:inline-flex; gap: 2px; }
.shell-C .knobs .seg .s {
  padding: 2px 7px; color: var(--muted); cursor: pointer;
  border: 1px solid var(--line2);
}
.shell-C .knobs .seg .s.on { color: var(--ink); background: var(--accent); border-color: var(--accent); }

.shell-C .caption {
  display:grid; grid-template-columns: 1fr auto auto auto;
  align-items:center; gap: 18px;
  padding: 6px 22px; border-bottom: 1px solid var(--line);
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); background: var(--surface);
}
.shell-C .caption b { color: var(--frost); font-weight: 400; }
.shell-C .caption .sort { color: var(--accent); }

.shell-C .results { flex: 1; overflow-y: auto; }
.shell-C .row {
  display: grid;
  grid-template-columns: 36px 88px 1fr 72px 36px;
  gap: 12px; align-items: center;
  padding: 8px 22px;
  border-bottom: 1px solid var(--line);
  cursor: pointer;
  position: relative;
}
.shell-C .row:hover { background: var(--surface); }
.shell-C .row.selected { background: var(--surface); }
.shell-C .row.selected::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--accent);
}
.shell-C .row .idx {
  font-size: 10.5px; color: var(--faint);
  font-variant-numeric: tabular-nums;
}
.shell-C .row .thumb {
  width: 88px; height: 66px;
  background: var(--pane) center/cover no-repeat;
  filter: contrast(1.04) brightness(0.95);
}
.shell-C .row .body { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.shell-C .row .body .titleline {
  display: flex; align-items: baseline; gap: 10px; min-width: 0;
}
.shell-C .row .body .film {
  font-family: var(--serif); font-size: 14px; color: var(--frost);
  letter-spacing: -0.005em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.shell-C .row .body .film .yr {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  margin-left: 6px;
}
.shell-C .row .body .ids {
  font-size: 10.5px; color: var(--accent); white-space: nowrap;
}
.shell-C .row .body .desc {
  font-size: 11.5px; color: var(--frost-dim); line-height: 1.4;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-family: var(--sans);
}
.shell-C .row .body .tags {
  display:flex; gap: 4px; flex-wrap:wrap;
  font-size: 10px; color: var(--faint);
}
.shell-C .row .body .tags .t { padding: 0; }
.shell-C .row .body .tags .t.m { color: var(--accent); }
.shell-C .row .body .tags .t + .t::before { content: ' '; }
.shell-C .row .score {
  font-size: 12px; color: var(--frost); text-align: right;
  display: flex; flex-direction: column; align-items: flex-end; gap: 2px;
}
.shell-C .row .score .bar {
  height: 2px; width: 60px; background: var(--line2); position: relative;
}
.shell-C .row .score .bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p, 80%); background: var(--accent);
}
.shell-C .row .chev { color: var(--faint); font-size: 12px; text-align: right; }

/* RIGHT PANE — inspector */
.shell-C .rp {
  border-left: 1px solid var(--line);
  display:flex; flex-direction:column; overflow:hidden; background: var(--pane);
}
.shell-C .rp .head {
  display:flex; align-items:center; justify-content:space-between;
  padding: 8px 14px; border-bottom: 1px solid var(--line);
  font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted);
}
.shell-C .rp .head .ct { color: var(--accent); }
.shell-C .rp .body { padding: 14px 16px; overflow-y: auto; flex: 1; }
.shell-C .rp .insp-kf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--ink) center/cover no-repeat;
  border: 1px solid var(--line2);
  filter: contrast(1.05) brightness(0.95);
}
.shell-C .rp .insp-film {
  font-family: var(--serif); font-size: 18px; color: var(--frost);
  margin-top: 12px; letter-spacing: -0.005em;
}
.shell-C .rp .insp-film .yr {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  margin-left: 6px;
}
.shell-C .rp .insp-meta {
  font-size: 11px; color: var(--muted); margin-top: 4px;
}
.shell-C .rp .insp-meta .v { color: var(--frost); }
.shell-C .rp .insp-section {
  font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); margin-top: 18px; padding-bottom: 4px;
  border-bottom: 1px solid var(--line);
}
.shell-C .rp .insp-desc {
  font-family: var(--sans); font-size: 12.5px; color: var(--frost-dim);
  line-height: 1.5; margin-top: 8px; text-wrap: pretty;
}
.shell-C .rp .signals { margin-top: 10px; display:grid; gap: 6px; }
.shell-C .rp .sig {
  display:grid; grid-template-columns: 70px 1fr 42px;
  align-items: center; gap: 10px;
  font-size: 11px; color: var(--frost-dim);
}
.shell-C .rp .sig .lab { color: var(--muted); }
.shell-C .rp .sig .track { height: 4px; background: var(--line2); position: relative; }
.shell-C .rp .sig .track::before {
  content:''; position:absolute; left:0; top:0; bottom:0;
  width: var(--p); background: var(--accent);
}
.shell-C .rp .sig .v { color: var(--frost); text-align: right; }
.shell-C .rp .sig.fused .track::before { background: var(--good); }
.shell-C .rp .insp-tags { display:flex; flex-wrap:wrap; gap: 4px 6px; margin-top: 10px; }
.shell-C .rp .insp-tags .t {
  font-size: 10.5px; color: var(--frost-dim);
  border: 1px solid var(--line2); padding: 2px 7px;
}
.shell-C .rp .insp-tags .t.m { color: var(--accent); border-color: var(--accent-dim); }
.shell-C .rp .actions {
  display:flex; flex-direction:column; gap: 4px; margin-top: 14px;
}
.shell-C .rp .actions .a {
  display:flex; align-items:center; justify-content:space-between;
  padding: 7px 10px; border: 1px solid var(--line2);
  font-size: 11px; cursor:pointer; color: var(--frost-dim);
}
.shell-C .rp .actions .a:hover { border-color: var(--accent); color: var(--frost); }
.shell-C .rp .actions .a .key { color: var(--muted); font-size: 10px; }
.shell-C .rp .actions .a.primary {
  background: var(--surface); color: var(--frost); border-color: var(--accent-dim);
}

/* BOTTOM STATUS */
.shell-C .botstatus {
  display:flex; align-items:center; justify-content:space-between;
  padding: 0 14px; border-top: 1px solid var(--line);
  background: var(--pane);
  font-size: 10.5px; color: var(--muted);
}
.shell-C .botstatus .mode {
  background: var(--accent); color: var(--ink); padding: 0 8px; height: 18px;
  display: inline-flex; align-items: center; margin-right: 12px;
  font-weight: 500; letter-spacing: 0.08em; font-size: 10px;
}
.shell-C .botstatus .keys { display:flex; align-items:center; gap: 14px; }
.shell-C .botstatus .keys .k { color: var(--frost-dim); }
.shell-C .botstatus .keys .k b { color: var(--accent); font-weight: 400; }
.shell-C .botstatus .right { display:flex; align-items:center; gap: 18px; }
.shell-C .botstatus .right b { color: var(--frost); font-weight: 400; }
.shell-C .botstatus .right .ok { color: var(--good); }

/* FOOT SYSTEM STRIP */
.shell-C .syslip {
  display: none;
}
`;

function ShellEstacao() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const filmsById = Object.fromEntries(films.map(f => [f.id, f]));
  const [sel, setSel] = React.useState(0);

  React.useEffect(() => {
    if (!document.getElementById('shell-C-css')) {
      const s = document.createElement('style');
      s.id = 'shell-C-css';
      s.textContent = SC_CSS;
      document.head.appendChild(s);
    }
  }, []);

  const r = results[sel];
  const f = filmsById[r.film];

  // Fake hybrid retrieval signals derived from score
  const sem  = Math.min(0.95, r.score + 0.05);
  const bm25 = Math.max(0.05, r.score - 0.55);
  const rk   = Math.min(0.95, r.score);
  const fused= r.score;

  return (
    <div className="shell-C">
      {/* TOP STATUS */}
      <div className="topstatus">
        <div className="left">
          <span className="brand"><span className="glyph">▣</span> cinemateca / mojica</span>
          <span className="crumb">
            acervo <span className="sep">›</span>
            <span className="cur">buscar</span> <span className="sep">·</span>
            text:semantic <span className="sep">·</span>
            scope:* <span className="sep">·</span>
            top_k:9
          </span>
        </div>
        <div className="right">
          <span><span className="dot"></span>idx ok</span>
          <span><span className="dot warn"></span>gpu:cuda:0</span>
          <span>1.588 cenas</span>
          <span>06 filmes</span>
          <span>v1.0.0</span>
          <span>PT</span>
        </div>
      </div>

      {/* TABLINE */}
      <div className="tabline">
        <div className="tab active"><span className="k">⌘1</span> buscar</div>
        <div className="tab"><span className="k">⌘2</span> cenas</div>
        <div className="tab"><span className="k">⌘3</span> anotar</div>
        <div className="tab"><span className="k">⌘4</span> rimas-visuais</div>
        <div className="tab"><span className="k">⌘5</span> proc <span className="pip warn">1</span></div>
        <div className="grow"></div>
        <div className="modes">
          <span>mode <b>SEARCH</b></span>
          <span>scope <b>acervo:*</b></span>
        </div>
      </div>

      {/* BODY */}
      <div className="body">

        {/* LEFT PANE */}
        <aside className="lp">
          <div className="head"><span>acervo · ./films</span><span className="ct">06</span></div>
          <div className="tree">
            <div className="tree-row open active">
              <span className="arrow">▾</span>
              <span>jeca_tatu</span>
              <span className="year">1959</span>
              <span className="ct">412</span>
            </div>
            <div className="tree-row indent">
              <span className="arrow">✓</span><span>frames</span><span></span><span className="ct">14,892</span>
            </div>
            <div className="tree-row indent">
              <span className="arrow">✓</span><span>cenas</span><span></span><span className="ct">412</span>
            </div>
            <div className="tree-row indent">
              <span className="arrow">✓</span><span>embeddings · CLIP</span><span></span><span className="ct">412</span>
            </div>
            <div className="tree-row indent">
              <span className="arrow">✓</span><span>descricoes · md2</span><span></span><span className="ct">412</span>
            </div>
            <div className="tree-row indent">
              <span className="arrow">✓</span><span>whisper · tr-pt</span><span></span><span className="ct">96m</span>
            </div>

            <div className="tree-row">
              <span className="arrow">▸</span><span>limite</span>
              <span className="year">1931</span><span className="ct">187</span>
            </div>
            <div className="tree-row">
              <span className="arrow">▸</span><span>rio_40_graus</span>
              <span className="year">1955</span><span className="ct">263</span>
            </div>
            <div className="tree-row">
              <span className="arrow">▸</span><span>o_cangaceiro</span>
              <span className="year">1953</span><span className="ct">298</span>
            </div>
            <div className="tree-row proc">
              <span className="arrow">⟳</span><span>aruanda</span>
              <span className="year">1960</span><span className="ct">94 · 78%</span>
            </div>
            <div className="tree-row">
              <span className="arrow">▸</span><span>o_pagador_de_promessas</span>
              <span className="year">1962</span><span className="ct">334</span>
            </div>
          </div>
          <div className="footer">
            <span>filmes</span><span className="v">06</span>
            <span>cenas</span><span className="v">1.588</span>
            <span>runtime</span><span className="v">8h 54m</span>
            <span>embeddings</span><span className="v">CLIP-L/14</span>
            <span>llm</span><span className="v">moondream-2</span>
          </div>
        </aside>

        {/* CENTER PANE */}
        <section className="cp">
          <div className="cmdline">
            <div className="prompt">
              <span className="pre">buscar ›</span>
              <input className="q" defaultValue="duas pessoas conversando ao ar livre" />
              <button className="submit">run ⏎</button>
            </div>
            <div className="knobs">
              <span><span className="k">modo:</span>
                <span className="seg">
                  <span className="s on">texto</span>
                  <span className="s">img</span>
                  <span className="s">audio</span>
                  <span className="s">multi</span>
                </span>
              </span>
              <span><span className="k">filme:</span><span className="v">*</span></span>
              <span><span className="k">hybrid:</span><span className="v">sem 0.70 · bm25 0.30</span></span>
              <span><span className="k">rerank:</span><span className="v">on (mxbai-rerank-l)</span></span>
              <span><span className="k">mmr:</span><span className="v">λ 0.50</span></span>
              <span><span className="k">k:</span><span className="v">9</span></span>
            </div>
          </div>

          <div className="caption">
            <span><b>009</b> results · 231 ms · sem ⊕ bm25 ⊕ rerank</span>
            <span className="sort">↓ score</span>
            <span>view · rows</span>
            <span>⌘G grid</span>
          </div>

          <div className="results">
            {results.map((rr, i) => {
              const ff = filmsById[rr.film];
              return (
                <div key={rr.id}
                     className={'row' + (i===sel ? ' selected' : '')}
                     onClick={() => setSel(i)}>
                  <span className="idx">{String(i+1).padStart(3,'0')}</span>
                  <span className="thumb" style={{backgroundImage:`url(${rr.kf})`}}></span>
                  <div className="body">
                    <div className="titleline">
                      <span className="film">{ff.title}<span className="yr">{ff.year}</span></span>
                      <span className="ids">cena {String(rr.cena).padStart(3,'0')} · {rr.tc}</span>
                    </div>
                    <span className="desc">{rr.desc}</span>
                    <span className="tags">
                      {rr.tags.map((t,j) => (
                        <span key={j} className={'t' + (t==='duas-pessoas'||t==='exterior' ? ' m' : '')}>{t}{j<rr.tags.length-1 ? ' ·' : ''}</span>
                      ))}
                    </span>
                  </div>
                  <span className="score">
                    {rr.score.toFixed(3)}
                    <span className="bar" style={{'--p': `${(rr.score*100).toFixed(0)}%`}}></span>
                  </span>
                  <span className="chev">›</span>
                </div>
              );
            })}
          </div>
        </section>

        {/* RIGHT PANE — inspector */}
        <aside className="rp">
          <div className="head"><span>inspector · ./scene</span><span className="ct">{String(sel+1).padStart(3,'0')}/009</span></div>
          <div className="body">
            <div className="insp-kf" style={{backgroundImage:`url(${r.kf})`}}></div>
            <div className="insp-film">{f.title}<span className="yr">{f.year}</span></div>
            <div className="insp-meta">
              <span className="v">{f.director}</span> · cena {String(r.cena).padStart(3,'0')} · {r.tc} · runtime {f.runtime}m
            </div>

            <div className="insp-section">why this result</div>
            <div className="signals">
              <div className="sig"><span className="lab">semantic</span>
                <span className="track" style={{'--p': `${(sem*100).toFixed(0)}%`}}></span>
                <span className="v">{sem.toFixed(3)}</span>
              </div>
              <div className="sig"><span className="lab">bm25</span>
                <span className="track" style={{'--p': `${(bm25*100).toFixed(0)}%`}}></span>
                <span className="v">{bm25.toFixed(3)}</span>
              </div>
              <div className="sig"><span className="lab">rerank</span>
                <span className="track" style={{'--p': `${(rk*100).toFixed(0)}%`}}></span>
                <span className="v">{rk.toFixed(3)}</span>
              </div>
              <div className="sig fused"><span className="lab">fused</span>
                <span className="track" style={{'--p': `${(fused*100).toFixed(0)}%`}}></span>
                <span className="v">{fused.toFixed(3)}</span>
              </div>
            </div>

            <div className="insp-section">description (md2)</div>
            <p className="insp-desc">{r.desc}</p>

            <div className="insp-section">tags</div>
            <div className="insp-tags">
              {r.tags.map((t,i) => (
                <span key={i} className={'t' + (t==='duas-pessoas'||t==='exterior' ? ' m' : '')}>{t}</span>
              ))}
            </div>

            <div className="actions">
              <div className="a primary"><span>open scene</span><span className="key">⏎</span></div>
              <div className="a"><span>find visual rhymes</span><span className="key">⌥R</span></div>
              <div className="a"><span>annotate · add tag</span><span className="key">A</span></div>
              <div className="a"><span>copy timecode</span><span className="key">⌘C</span></div>
            </div>
          </div>
        </aside>

      </div>

      {/* BOTTOM STATUS */}
      <div className="botstatus">
        <div style={{display:'flex',alignItems:'center'}}>
          <span className="mode">NORMAL</span>
          <div className="keys">
            <span className="k"><b>j/k</b> nav</span>
            <span className="k"><b>⏎</b> open</span>
            <span className="k"><b>^F</b> search</span>
            <span className="k"><b>^K</b> cmd</span>
            <span className="k"><b>⌘1-5</b> tabs</span>
            <span className="k"><b>⌥R</b> rhymes</span>
          </div>
        </div>
        <div className="right">
          <span>row <b>{String(sel+1).padStart(3,'0')}</b> / 009</span>
          <span><span className="ok">●</span> idx ok</span>
          <span>231 ms</span>
          <span>v1.0.0</span>
          <span>UTF-8</span>
          <span>pt-BR</span>
        </div>
      </div>
    </div>
  );
}

window.ShellEstacao = ShellEstacao;
