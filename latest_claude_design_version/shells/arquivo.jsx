// Shell A — ARQUIVO
// Editorial cinematheque. Slim left rail as printed program, restrained
// vermilion accent, Newsreader display + Geist body + JetBrains Mono.

const SA_PALETTE = {
  ink:        '#0E1014',
  surface:    '#15181D',
  raised:     '#1B1E24',
  hairline:   '#2A2D34',
  hairline2:  '#3B3E46',
  paper:      '#ECE6D8',
  paperDim:   '#B8B0A0',
  muted:      '#7D7568',
  faint:      '#54504A',
  accent:     '#DC462A',   // Brazilian vermilion
  accentDim:  '#8C2E1B',
  gold:       '#C7A55C',
};

const SA_CSS = `
.shell-A { all: initial; box-sizing: border-box; }
.shell-A *, .shell-A *::before, .shell-A *::after { box-sizing: border-box; }
.shell-A {
  --ink: ${SA_PALETTE.ink};
  --surface: ${SA_PALETTE.surface};
  --raised: ${SA_PALETTE.raised};
  --line: ${SA_PALETTE.hairline};
  --line2: ${SA_PALETTE.hairline2};
  --paper: ${SA_PALETTE.paper};
  --paper-dim: ${SA_PALETTE.paperDim};
  --muted: ${SA_PALETTE.muted};
  --faint: ${SA_PALETTE.faint};
  --accent: ${SA_PALETTE.accent};
  --accent-dim: ${SA_PALETTE.accentDim};
  --gold: ${SA_PALETTE.gold};
  --serif: 'Newsreader', 'Source Serif 4', Georgia, serif;
  --sans: 'Geist', system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Courier New', monospace;
  display: grid; grid-template-columns: 268px 1fr; height: 100%; width: 100%;
  background: var(--ink); color: var(--paper);
  font-family: var(--sans); font-size: 13px; line-height: 1.55;
  letter-spacing: -0.005em;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'ss01' on, 'cv11' on;
}

/* ─── RAIL ─────────────────────────────────────────────────────────── */
.shell-A .rail {
  border-right: 1px solid var(--line);
  display: flex; flex-direction: column;
  padding: 22px 0 18px;
}
.shell-A .rail-brand { padding: 0 22px 22px; }
.shell-A .rail-brand .mark { display:flex; align-items:center; gap:9px; }
.shell-A .rail-brand .mark svg { display:block; }
.shell-A .rail-brand .name {
  font-family: var(--serif); font-weight: 400; font-size: 16.5px;
  letter-spacing: -0.01em; line-height: 1.15; color: var(--paper);
}
.shell-A .rail-brand .sub {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); margin-top: 4px;
}
.shell-A .rail-section {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 14px 22px 10px;
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--muted);
  border-top: 1px solid var(--line);
}
.shell-A .rail-section .count { color: var(--faint); }
.shell-A .rail-films { padding: 4px 14px 14px; }
.shell-A .film {
  display:grid; grid-template-columns: 30px 1fr auto; align-items:baseline;
  padding: 7px 8px; border-radius: 3px; cursor: pointer; position: relative;
  column-gap: 6px;
}
.shell-A .film:hover { background: rgba(220,70,42,0.04); }
.shell-A .film.active { background: rgba(220,70,42,0.06); }
.shell-A .film.active::before {
  content:''; position:absolute; left:-1px; top:8px; bottom:8px; width:2px;
  background: var(--accent);
}
.shell-A .film-year {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.04em; font-variant-numeric: tabular-nums;
}
.shell-A .film-title {
  font-family: var(--serif); font-weight: 400; font-size: 14px;
  color: var(--paper-dim); line-height: 1.2;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.shell-A .film.active .film-title { color: var(--paper); }
.shell-A .film-scenes {
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  font-variant-numeric: tabular-nums;
}
.shell-A .film.processing .film-title { color: var(--gold); font-style: italic; }
.shell-A .film.processing .film-scenes { color: var(--gold); }

.shell-A .rail-meta {
  margin-top: auto; padding: 14px 22px 0;
  border-top: 1px solid var(--line);
  display: flex; flex-direction: column; gap: 10px;
}
.shell-A .rail-stats {
  display:grid; grid-template-columns: 1fr 1fr; gap: 6px 14px;
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.shell-A .rail-stats .v { color: var(--paper); }
.shell-A .rail-footrow {
  display:flex; align-items:center; justify-content:space-between;
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--faint);
}
.shell-A .rail-footrow a { color: var(--paper-dim); text-decoration: none; cursor: pointer; }
.shell-A .rail-footrow a:hover { color: var(--accent); }
.shell-A .locale .on { color: var(--paper); }

/* ─── MAIN ─────────────────────────────────────────────────────────── */
.shell-A .main { display: flex; flex-direction: column; overflow: hidden; }

.shell-A .topbar {
  padding: 20px 36px 0; display:flex; align-items:center; justify-content:space-between;
  border-bottom: 1px solid var(--line);
}
.shell-A .nav { display:flex; align-items:baseline; gap: 0; padding-bottom: 16px; }
.shell-A .nav-item {
  font-family: var(--sans); font-size: 12.5px; color: var(--muted);
  padding: 0 14px 0 12px; cursor: pointer; display:flex; align-items:center;
  gap: 7px; position: relative; letter-spacing: -0.005em;
}
.shell-A .nav-item + .nav-item { border-left: 1px solid var(--line); }
.shell-A .nav-item .dot {
  width: 4px; height: 4px; border-radius: 50%; background: transparent;
}
.shell-A .nav-item.active { color: var(--paper); }
.shell-A .nav-item.active .dot { background: var(--accent); }
.shell-A .nav-item .badge {
  font-family: var(--mono); font-size: 9.5px; padding: 1px 5px; color: var(--gold);
  border: 1px solid var(--line2); border-radius: 9px; margin-left: 4px;
  font-variant-numeric: tabular-nums;
}
.shell-A .breadcrumb {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); padding-bottom: 16px;
}
.shell-A .breadcrumb .sep { color: var(--faint); margin: 0 8px; }

/* ─── SEARCH ────────────────────────────────────────────────────────── */
.shell-A .search-block { padding: 30px 36px 18px; }
.shell-A .search-row {
  display:flex; align-items:center; gap: 18px;
  padding: 14px 0 18px;
  border-bottom: 1px solid var(--line);
}
.shell-A .search-row .q {
  flex: 1; font-family: var(--mono); font-size: 22px;
  background: transparent; border: none; outline: none;
  color: var(--paper); letter-spacing: -0.01em;
  caret-color: var(--accent);
}
.shell-A .search-row .q::placeholder { color: var(--faint); font-family: var(--serif); font-style: italic; }
.shell-A .search-row .submit {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  display:flex; align-items:center; gap:8px;
  letter-spacing: 0.12em; text-transform: uppercase;
  border: 1px solid var(--line2); padding: 6px 10px; border-radius: 2px;
  cursor:pointer; background: transparent;
}
.shell-A .search-row .submit:hover { color: var(--accent); border-color: var(--accent-dim); }

.shell-A .mode-row {
  display:flex; align-items:center; justify-content:space-between; gap: 24px;
  padding-top: 14px;
}
.shell-A .modes { display:flex; gap: 22px; }
.shell-A .mode {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); cursor: pointer;
  display:flex; align-items:center; gap:7px;
}
.shell-A .mode .glyph { color: var(--faint); }
.shell-A .mode.active { color: var(--paper); }
.shell-A .mode.active .glyph { color: var(--accent); }
.shell-A .knobs { display:flex; gap: 18px; align-items:center; }
.shell-A .knob {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  display:flex; align-items:center; gap:6px; letter-spacing: 0.04em;
}
.shell-A .knob .k { color: var(--faint); text-transform: uppercase; letter-spacing: 0.16em; }
.shell-A .knob .v { color: var(--paper-dim); font-variant-numeric: tabular-nums; }

/* ─── CAPTION ───────────────────────────────────────────────────────── */
.shell-A .caption {
  padding: 20px 36px 12px;
  display: grid; grid-template-columns: auto 1fr auto; gap: 18px; align-items: baseline;
}
.shell-A .caption .head {
  font-family: var(--serif); font-style: italic; font-size: 17px; color: var(--paper);
  letter-spacing: -0.01em;
}
.shell-A .caption .head em { color: var(--accent); font-style: italic; font-weight: 400; }
.shell-A .caption .ord {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
}

/* ─── GRID ──────────────────────────────────────────────────────────── */
.shell-A .grid {
  padding: 16px 36px 36px;
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  column-gap: 28px; row-gap: 28px;
  overflow-y: auto; flex: 1;
}
.shell-A .scene { display: flex; flex-direction: column; gap: 10px; }
.shell-A .scene .kf {
  width: 100%; aspect-ratio: 4/3; background: var(--ink) center/cover no-repeat;
  filter: contrast(1.04) brightness(0.96);
}
.shell-A .scene .meta-top {
  display:flex; align-items: baseline; justify-content: space-between;
  padding: 4px 0 6px; border-bottom: 1px solid var(--line);
}
.shell-A .scene .film-attr {
  font-family: var(--serif); font-style: italic; font-size: 14.5px;
  color: var(--paper); letter-spacing: -0.01em;
}
.shell-A .scene .film-attr .yr {
  font-family: var(--mono); font-style: normal; color: var(--muted);
  font-size: 10.5px; margin-left: 6px; letter-spacing: 0.04em;
}
.shell-A .scene .score {
  font-family: var(--mono); font-size: 11px; color: var(--gold);
  font-variant-numeric: tabular-nums;
}
.shell-A .scene .ids {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted);
}
.shell-A .scene .desc {
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  color: var(--paper-dim); text-wrap: pretty;
}
.shell-A .scene .tags {
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  letter-spacing: 0.06em;
}
.shell-A .scene .tags .t + .t::before {
  content: ' · '; color: var(--faint); margin: 0 1px;
}
.shell-A .scene .tags .t.matched { color: var(--accent); }

/* ─── FOOT STRIP (system mini) ───────────────────────────────────────── */
.shell-A .syslip {
  border-top: 1px solid var(--line);
  padding: 14px 36px 16px;
  display: grid; grid-template-columns: 1fr auto;
  gap: 24px; align-items:center;
  background: var(--surface);
}
.shell-A .swatches { display:flex; gap: 0; align-items:center; }
.shell-A .swatch {
  display:flex; flex-direction:column; gap: 4px; padding-right: 12px;
}
.shell-A .swatch .chip { width: 28px; height: 28px; border: 1px solid var(--line2); }
.shell-A .swatch .lab {
  font-family: var(--mono); font-size: 8.5px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.shell-A .syslip .system-label {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--faint);
}
`;

function SAMark() {
  // Small institutional mark: a film-strip fragment.
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <rect x="0.5" y="2.5" width="21" height="17" stroke={SA_PALETTE.paper} strokeWidth="1"/>
      <rect x="2"   y="4"   width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="2"   y="9.5" width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="2"   y="15"  width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="17"  y="4"   width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="17"  y="9.5" width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="17"  y="15"  width="3" height="3" fill={SA_PALETTE.paper}/>
      <rect x="7"   y="5"   width="8" height="12" fill={SA_PALETTE.accent}/>
    </svg>
  );
}

function ShellArquivo() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const filmsById = Object.fromEntries(films.map(f => [f.id, f]));

  React.useEffect(() => {
    if (!document.getElementById('shell-A-css')) {
      const s = document.createElement('style');
      s.id = 'shell-A-css';
      s.textContent = SA_CSS;
      document.head.appendChild(s);
    }
  }, []);

  return (
    <div className="shell-A">
      {/* RAIL */}
      <aside className="rail">
        <div className="rail-brand">
          <div className="mark">
            <SAMark />
            <div>
              <div className="name">Cinemateca Mojica</div>
              <div className="sub">Acervo Digital · v1.0</div>
            </div>
          </div>
        </div>

        <div className="rail-section">
          <span>Programa · 2026</span>
          <span className="count">06 / 06</span>
        </div>
        <div className="rail-films">
          {films.map((f, i) => (
            <div key={f.id} className={'film' + (i===0 ? ' active' : '') + (i===4 ? ' processing' : '')}>
              <span className="film-year">{f.year}</span>
              <span className="film-title">{f.title}</span>
              <span className="film-scenes">{f.scenes}</span>
            </div>
          ))}
        </div>

        <div className="rail-meta">
          <div className="rail-stats">
            <span>FILMES</span><span className="v">06</span>
            <span>CENAS</span><span className="v">1.588</span>
            <span>HORAS</span><span className="v">8.9</span>
            <span>EMBEDDINGS</span><span className="v">CLIP · MDR</span>
          </div>
          <div className="rail-footrow">
            <span className="locale"><span className="on">PT</span> · EN</span>
            <span><a>Sobre</a></span>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="main">
        <div className="topbar">
          <nav className="nav">
            <span className="nav-item active"><span className="dot"></span>Buscar</span>
            <span className="nav-item"><span className="dot"></span>Cenas</span>
            <span className="nav-item"><span className="dot"></span>Anotar</span>
            <span className="nav-item"><span className="dot"></span>Rimas visuais</span>
            <span className="nav-item"><span className="dot"></span>Processamento <span className="badge">1</span></span>
          </nav>
          <div className="breadcrumb">
            Acervo <span className="sep">/</span> Busca semântica
          </div>
        </div>

        <div className="search-block">
          <div className="search-row">
            <input className="q" defaultValue="duas pessoas conversando ao ar livre" />
            <button className="submit">Buscar <span>→</span></button>
          </div>
          <div className="mode-row">
            <div className="modes">
              <span className="mode active"><span className="glyph">●</span>Texto</span>
              <span className="mode"><span className="glyph">○</span>Imagem</span>
              <span className="mode"><span className="glyph">○</span>Trilha</span>
              <span className="mode"><span className="glyph">○</span>Multimodal</span>
            </div>
            <div className="knobs">
              <span className="knob"><span className="k">Acervo</span><span className="v">inteiro</span></span>
              <span className="knob"><span className="k">Híbrido</span><span className="v">sem · 0.7  bm25 · 0.3</span></span>
              <span className="knob"><span className="k">Rerank</span><span className="v">on</span></span>
              <span className="knob"><span className="k">MMR</span><span className="v">λ 0.5</span></span>
            </div>
          </div>
        </div>

        <div className="caption">
          <span className="ord">009 resultados</span>
          <span className="head">de seis filmes · semântico <em>⊕</em> BM25 <em>⊕</em> cross-encoder</span>
          <span className="ord">231 ms</span>
        </div>

        <div className="grid">
          {results.map(r => {
            const f = filmsById[r.film];
            return (
              <article key={r.id} className="scene">
                <div className="kf" style={{backgroundImage:`url(${r.kf})`}} />
                <div className="meta-top">
                  <span className="film-attr">{f.title}<span className="yr">{f.year}</span></span>
                  <span className="score">{r.score.toFixed(3)}</span>
                </div>
                <span className="ids">cena {String(r.cena).padStart(3,'0')} · {r.tc}</span>
                <p className="desc">{r.desc}</p>
                <div className="tags">
                  {r.tags.map((t,i) => (
                    <span key={i} className={'t' + (t==='duas-pessoas' || t==='exterior' ? ' matched' : '')}>{t}</span>
                  ))}
                </div>
              </article>
            );
          })}
        </div>

        <div className="syslip">
          <div className="swatches">
            {[
              ['ink','#0E1014'], ['surface','#15181D'], ['line','#2A2D34'],
              ['paper','#ECE6D8'], ['muted','#7D7568'],
              ['accent','#DC462A'], ['gold','#C7A55C'],
            ].map(([n,c]) => (
              <div key={n} className="swatch">
                <div className="chip" style={{background:c}}></div>
                <div className="lab">{n}</div>
              </div>
            ))}
          </div>
          <div className="system-label">
            Newsreader · Geist · JetBrains Mono
          </div>
        </div>
      </main>
    </div>
  );
}

window.ShellArquivo = ShellArquivo;
