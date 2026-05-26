// Shell B — CINETECA
// Festival / bold editorial. Oversized italic display (Instrument Serif),
// Schibsted Grotesk for body, single acid-lime accent. No fixed sidebar —
// the catalogue lives in a drawer chip and the masthead is the orientation.

const SB_PALETTE = {
  ink:        '#0A0908',
  surface:    '#14120F',
  raised:     '#1B1815',
  hairline:   '#26221C',
  hairline2:  '#3A352C',
  bone:       '#F4EFE5',
  boneDim:    '#C8C0AE',
  muted:      '#86806F',
  faint:      '#4F4A41',
  accent:     '#D7E839',  // acid lime / chartreuse
  accentDim:  '#97A41E',
  ember:      '#E07A2B',
};

const SB_CSS = `
.shell-B { all: initial; box-sizing: border-box; }
.shell-B *, .shell-B *::before, .shell-B *::after { box-sizing: border-box; }
.shell-B {
  --ink: ${SB_PALETTE.ink};
  --surface: ${SB_PALETTE.surface};
  --raised: ${SB_PALETTE.raised};
  --line: ${SB_PALETTE.hairline};
  --line2: ${SB_PALETTE.hairline2};
  --bone: ${SB_PALETTE.bone};
  --bone-dim: ${SB_PALETTE.boneDim};
  --muted: ${SB_PALETTE.muted};
  --faint: ${SB_PALETTE.faint};
  --accent: ${SB_PALETTE.accent};
  --accent-dim: ${SB_PALETTE.accentDim};
  --ember: ${SB_PALETTE.ember};
  --display: 'Instrument Serif', 'GT Sectra', Georgia, serif;
  --sans: 'Schibsted Grotesk', system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'Courier New', monospace;
  display: flex; flex-direction: column;
  height: 100%; width: 100%;
  background: var(--ink); color: var(--bone);
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* ─── MASTHEAD ──────────────────────────────────────────────────────── */
.shell-B .masthead {
  display: grid; grid-template-columns: auto 1fr auto;
  align-items: center; gap: 28px;
  padding: 14px 36px;
  border-bottom: 1px solid var(--line);
}
.shell-B .brand {
  display:flex; align-items:baseline; gap: 10px;
  font-family: var(--display); font-style: italic; font-weight: 400;
  font-size: 22px; line-height: 1; letter-spacing: -0.01em;
  color: var(--bone);
}
.shell-B .brand .dot { color: var(--accent); }
.shell-B .brand .sub {
  font-family: var(--mono); font-style: normal; font-size: 9.5px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
  padding-left: 12px; border-left: 1px solid var(--line2); margin-left: 4px;
}
.shell-B .mast-nav { display:flex; align-items:center; gap: 22px; justify-content:center; }
.shell-B .mast-nav .item {
  font-family: var(--sans); font-size: 12px; font-weight: 500;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); cursor: pointer; padding: 6px 0;
  position: relative;
}
.shell-B .mast-nav .item.active { color: var(--bone); }
.shell-B .mast-nav .item.active::after {
  content: ''; position: absolute; left: -8px; right: -8px; bottom: -2px;
  height: 2px; background: var(--accent);
}
.shell-B .mast-nav .item .pip {
  display: inline-block; min-width: 14px; padding: 0 4px; margin-left: 6px;
  font-family: var(--mono); font-size: 9.5px; font-weight: 400;
  background: var(--accent); color: var(--ink); letter-spacing: 0;
  text-transform: none; line-height: 14px; text-align: center;
}
.shell-B .mast-right {
  display:flex; align-items:center; gap: 18px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.shell-B .mast-right .locale .on { color: var(--bone); }
.shell-B .mast-right .ver { color: var(--faint); }
.shell-B .catalog-chip {
  display:flex; align-items:center; gap: 9px;
  font-family: var(--sans); font-size: 11.5px; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 7px 12px 7px 10px; cursor: pointer; color: var(--bone);
  border: 1px solid var(--line2);
}
.shell-B .catalog-chip:hover { border-color: var(--accent); }
.shell-B .catalog-chip .gly { font-family: var(--mono); color: var(--accent); }
.shell-B .catalog-chip .pip {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0; text-transform: none;
}

/* ─── HERO / SEARCH ─────────────────────────────────────────────────── */
.shell-B .hero { padding: 42px 36px 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: end; }
.shell-B .hero .title {
  font-family: var(--display); font-weight: 400;
  font-size: 130px; line-height: 0.88; letter-spacing: -0.03em;
  color: var(--bone);
}
.shell-B .hero .title .it { font-style: italic; }
.shell-B .hero .title .ac { color: var(--accent); }
.shell-B .hero .sub {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); margin-top: 18px;
}
.shell-B .hero .sub b { color: var(--bone); font-weight: 400; }

.shell-B .qzone {
  display: flex; flex-direction: column; gap: 16px;
  padding-bottom: 6px;
}
.shell-B .qrow { display:flex; align-items:flex-end; gap: 14px; border-bottom: 2px solid var(--bone); padding-bottom: 10px; }
.shell-B .qrow .q {
  flex: 1; background: transparent; border: none; outline: none;
  font-family: var(--display); font-style: italic; font-weight: 400;
  font-size: 30px; line-height: 1.1; color: var(--bone);
  letter-spacing: -0.01em;
  caret-color: var(--accent);
}
.shell-B .qrow .q::placeholder { color: var(--faint); }
.shell-B .qrow .qsubmit {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
  color: var(--ink); background: var(--accent); border: none;
  text-transform: uppercase; padding: 9px 14px; cursor: pointer;
  display:flex; align-items:center; gap: 8px;
}
.shell-B .qmodes { display:flex; gap: 10px; align-items:center; flex-wrap:wrap; }
.shell-B .qmodes .chip {
  font-family: var(--sans); font-size: 11.5px; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 7px 12px; cursor: pointer; color: var(--bone-dim);
  border: 1px solid var(--line2);
}
.shell-B .qmodes .chip.active {
  background: var(--accent); color: var(--ink); border-color: var(--accent);
}
.shell-B .qmodes .div {
  width: 1px; height: 18px; background: var(--line2); margin: 0 4px;
}
.shell-B .qmodes .knob {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.04em;
}
.shell-B .qmodes .knob b { color: var(--bone); font-weight: 400; font-variant-numeric: tabular-nums; }

/* ─── SECTION HEADER ────────────────────────────────────────────────── */
.shell-B .section-head {
  display:grid; grid-template-columns: auto 1fr auto;
  gap: 18px; align-items: baseline;
  padding: 18px 36px 12px;
  border-top: 1px solid var(--line);
  margin-top: 12px;
}
.shell-B .section-head .label {
  font-family: var(--display); font-style: italic; font-size: 20px; color: var(--bone);
}
.shell-B .section-head .meta {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.shell-B .section-head .meta b { color: var(--accent); font-weight: 400; }

/* ─── RESULTS GRID ──────────────────────────────────────────────────── */
.shell-B .results {
  padding: 6px 36px 24px;
  display: grid; grid-template-columns: 1.8fr 1fr 1fr;
  gap: 26px;
  flex: 1; overflow-y: auto;
}
.shell-B .card { display: flex; flex-direction: column; gap: 12px; }
.shell-B .card.hero { grid-row: span 2; }
.shell-B .card .kf {
  width: 100%; aspect-ratio: 4/3; background: var(--surface) center/cover no-repeat;
  position: relative; filter: contrast(1.05) brightness(0.97);
}
.shell-B .card.hero .kf { aspect-ratio: 16/10; }
.shell-B .card .kf .badge {
  position: absolute; top: 10px; left: 10px;
  font-family: var(--mono); font-size: 10px; padding: 3px 6px;
  background: var(--accent); color: var(--ink);
  letter-spacing: 0.08em; text-transform: uppercase;
}
.shell-B .card .kf .score {
  position: absolute; bottom: 10px; right: 10px;
  font-family: var(--mono); font-size: 11px; padding: 3px 6px;
  background: rgba(10,9,8,0.78); color: var(--accent);
  letter-spacing: 0.04em;
  font-variant-numeric: tabular-nums;
  backdrop-filter: blur(4px);
}
.shell-B .card .attr {
  display:flex; align-items: baseline; justify-content: space-between;
  gap: 10px;
}
.shell-B .card .film {
  font-family: var(--display); font-style: italic; font-weight: 400;
  font-size: 22px; line-height: 1.05; color: var(--bone); letter-spacing: -0.01em;
}
.shell-B .card.hero .film { font-size: 38px; }
.shell-B .card .film .yr {
  font-family: var(--mono); font-style: normal; color: var(--muted);
  font-size: 11px; letter-spacing: 0.04em; margin-left: 8px;
}
.shell-B .card .director {
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--faint);
}
.shell-B .card .ids {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.04em; font-variant-numeric: tabular-nums;
}
.shell-B .card .desc {
  font-family: var(--sans); font-size: 13.5px; color: var(--bone-dim);
  line-height: 1.45; text-wrap: pretty;
}
.shell-B .card.hero .desc { font-size: 15px; max-width: 46ch; }
.shell-B .card .tagrow { display:flex; gap: 6px; flex-wrap: wrap; }
.shell-B .card .tag {
  font-family: var(--mono); font-size: 10px;
  padding: 2px 7px; color: var(--bone-dim);
  border: 1px solid var(--line2);
  letter-spacing: 0.02em;
}
.shell-B .card .tag.matched { color: var(--accent); border-color: var(--accent-dim); }

/* ─── FOOT SYSTEM ───────────────────────────────────────────────────── */
.shell-B .syslip {
  border-top: 1px solid var(--line);
  padding: 14px 36px;
  display: grid; grid-template-columns: 1fr auto;
  gap: 24px; align-items:center; background: var(--surface);
}
.shell-B .swatches { display:flex; gap: 0; align-items:center; }
.shell-B .swatch { display:flex; flex-direction:column; gap: 4px; padding-right: 12px; }
.shell-B .swatch .chip { width: 28px; height: 28px; }
.shell-B .swatch .lab {
  font-family: var(--mono); font-size: 8.5px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.shell-B .syslip .system-label {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--faint);
}
`;

function ShellCineteca() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const filmsById = Object.fromEntries(films.map(f => [f.id, f]));
  const hero = results[0];
  const sideCol = results.slice(1, 3); // 2 cards
  const restRow = results.slice(3, 9); // 6 cards (3-col)

  React.useEffect(() => {
    if (!document.getElementById('shell-B-css')) {
      const s = document.createElement('style');
      s.id = 'shell-B-css';
      s.textContent = SB_CSS;
      document.head.appendChild(s);
    }
  }, []);

  const Card = ({r, big}) => {
    const f = filmsById[r.film];
    return (
      <article className={'card' + (big ? ' hero' : '')}>
        <div className="kf" style={{backgroundImage:`url(${r.kf})`}}>
          {big && <span className="badge">resultado · 1</span>}
          <span className="score">{r.score.toFixed(3)}</span>
        </div>
        <div className="attr">
          <span className="film">{f.title}<span className="yr">{f.year}</span></span>
        </div>
        <span className="director">{f.director} · cena {String(r.cena).padStart(3,'0')} · {r.tc}</span>
        <p className="desc">{r.desc}</p>
        <div className="tagrow">
          {r.tags.slice(0, big ? 6 : 4).map((t,i) => (
            <span key={i} className={'tag' + (t==='duas-pessoas'||t==='exterior' ? ' matched' : '')}>{t}</span>
          ))}
        </div>
      </article>
    );
  };

  return (
    <div className="shell-B">
      {/* MASTHEAD */}
      <header className="masthead">
        <div style={{display:'flex', alignItems:'center', gap: 16}}>
          <div className="brand">
            Cinemateca <span className="dot">·</span> Mojica
            <span className="sub">Acervo Digital · v1.0</span>
          </div>
        </div>
        <nav className="mast-nav">
          <span className="item active">Buscar</span>
          <span className="item">Cenas</span>
          <span className="item">Anotar</span>
          <span className="item">Rimas visuais</span>
          <span className="item">Processamento<span className="pip">1</span></span>
        </nav>
        <div className="mast-right">
          <button className="catalog-chip">
            <span className="gly">▤</span> Catálogo <span className="pip">06</span>
          </button>
          <span className="locale"><span className="on">PT</span>/EN</span>
        </div>
      </header>

      {/* HERO + SEARCH */}
      <section className="hero">
        <div>
          <div className="title">
            <span className="it">Buscar</span><span className="ac">.</span>
          </div>
          <div className="sub">no acervo · <b>1.588 cenas</b> indexadas · <b>06 filmes</b> · 1931 — 1962</div>
        </div>
        <div className="qzone">
          <div className="qrow">
            <input className="q" defaultValue="duas pessoas conversando ao ar livre" />
            <button className="qsubmit">Buscar ⏎</button>
          </div>
          <div className="qmodes">
            <span className="chip active">● Texto</span>
            <span className="chip">○ Imagem</span>
            <span className="chip">○ Trilha</span>
            <span className="chip">○ Multimodal</span>
            <span className="div"></span>
            <span className="knob">híbrido <b>sem 0.7 · bm25 0.3</b></span>
            <span className="knob">rerank <b>on</b></span>
            <span className="knob">MMR <b>λ 0.5</b></span>
          </div>
        </div>
      </section>

      {/* SECTION HEAD */}
      <div className="section-head">
        <span className="label">Cenas afins.</span>
        <span></span>
        <span className="meta"><b>009</b> resultados · 231 ms · cross-encoder rerank</span>
      </div>

      {/* RESULTS */}
      <div className="results">
        <Card r={hero} big />
        {sideCol.map(r => <Card key={r.id} r={r} />)}
        {restRow.map(r => <Card key={r.id} r={r} />)}
      </div>

      {/* FOOT */}
      <div className="syslip">
        <div className="swatches">
          {[
            ['ink','#0A0908'], ['surface','#14120F'], ['line','#3A352C'],
            ['bone','#F4EFE5'], ['muted','#86806F'],
            ['accent','#D7E839'], ['ember','#E07A2B'],
          ].map(([n,c]) => (
            <div key={n} className="swatch">
              <div className="chip" style={{background:c}}></div>
              <div className="lab">{n}</div>
            </div>
          ))}
        </div>
        <div className="system-label">
          Instrument Serif · Schibsted Grotesk · JetBrains Mono
        </div>
      </div>
    </div>
  );
}

window.ShellCineteca = ShellCineteca;
