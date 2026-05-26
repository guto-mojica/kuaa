// Mojica · Frame.io branch · Rimas Visuais (cross-film visual echoes)
// Signature feature. Anchor scene + gallery of visually-similar scenes
// from OTHER films, ordered by embedding similarity. Selecting an echo
// reveals "why" — anchor↔echo comparison with embedding signal bars.

const RIMAS_CSS = `
.r-cp { display: flex; flex-direction: column; min-width: 0; overflow: hidden; background: var(--bg); }

/* Top tool row */
.r-cp .topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 24px; border-bottom: 1px solid var(--bd);
  gap: 18px;
}
.r-cp .topbar .l {
  display: flex; align-items: center; gap: 14px;
}
.r-cp .topbar .l .h {
  display: flex; align-items: center; gap: 10px;
  font-size: 16px; font-weight: 600; color: var(--t);
  letter-spacing: -0.01em;
}
.r-cp .topbar .l .h .ic {
  width: 26px; height: 26px; border-radius: 6px;
  background: var(--ac-bg); color: var(--ac);
  display: flex; align-items: center; justify-content: center;
}
.r-cp .topbar .l .ct {
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  padding: 2px 8px; background: var(--raised); border-radius: 9px;
}
.r-cp .topbar .l .ct b { color: var(--ac); font-weight: 600; }
.r-cp .topbar .r { display: flex; align-items: center; gap: 8px; }

.r-cp .controls {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 10px 24px; border-bottom: 1px solid var(--bd); background: var(--bg);
}
.r-cp .controls .knob {
  display: flex; align-items: center; gap: 7px;
  font-size: 11.5px; color: var(--muted);
}
.r-cp .controls .knob .k {
  text-transform: uppercase; letter-spacing: 0.06em;
  font-size: 10.5px; color: var(--faint); font-family: var(--mono);
}
.r-cp .controls .knob .v {
  color: var(--t2); font-family: var(--mono); font-size: 11px;
  padding: 2px 7px; background: var(--panel); border: 1px solid var(--bd);
  border-radius: 4px;
}
.r-cp .controls .knob .v.acc { color: var(--ac); border-color: var(--ac-dim); }
.r-cp .controls .div { width: 1px; height: 20px; background: var(--bd); margin: 0 4px; }
.r-cp .controls .grow { flex: 1; }
.r-cp .controls .modes {
  display: flex; align-items: center; gap: 4px;
  background: var(--panel); border: 1px solid var(--bd); border-radius: 6px; padding: 2px;
}
.r-cp .controls .modes .m {
  padding: 4px 9px; border-radius: 4px;
  font-size: 11.5px; color: var(--muted); cursor: pointer; font-weight: 500;
}
.r-cp .controls .modes .m.on { background: var(--hover); color: var(--t); }

/* SCROLL */
.r-cp .scroll {
  flex: 1; overflow-y: auto;
}

/* ANCHOR SECTION */
.r-anchor {
  display: grid; grid-template-columns: minmax(0, 480px) 1fr;
  gap: 22px; padding: 22px 24px;
  border-bottom: 1px solid var(--bd);
  background: linear-gradient(180deg, rgba(139,123,216,0.04), transparent);
}
.r-anchor .kf-wrap { position: relative; }
.r-anchor .kf {
  width: 100%; aspect-ratio: 4/3;
  background-size: cover; background-position: center;
  background-color: var(--bg);
  border-radius: 8px; border: 1px solid var(--bd2);
  filter: contrast(1.04) brightness(0.97);
}
.r-anchor .kf-label {
  position: absolute; top: 12px; left: 12px;
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 14px;
  background: var(--ac); color: #fff;
  font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
  box-shadow: 0 0 0 3px rgba(139,123,216,0.22);
}
.r-anchor .kf-label .gly {
  width: 6px; height: 6px; border-radius: 50%; background: #fff;
}
.r-anchor .meta { display: flex; flex-direction: column; gap: 10px; padding-top: 4px; }
.r-anchor .meta .ll {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.r-anchor .meta .ll .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ac); }
.r-anchor .meta h1 {
  margin: 0; font-size: 26px; font-weight: 700; color: var(--t);
  letter-spacing: -0.022em; line-height: 1.15;
}
.r-anchor .meta .filmrow {
  display: flex; align-items: center; gap: 10px;
  font-size: 13.5px; color: var(--t2);
}
.r-anchor .meta .filmrow .pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 14px;
  background: var(--raised); border: 1px solid var(--bd2);
}
.r-anchor .meta .filmrow .pill .dot { width: 7px; height: 7px; border-radius: 50%; }
.r-anchor .meta .filmrow .pill b { color: var(--t); font-weight: 600; }
.r-anchor .meta .ids {
  font-family: var(--mono); font-size: 12px; color: var(--muted);
  letter-spacing: 0.08em; display: flex; align-items: center; gap: 12px;
}
.r-anchor .meta .ids .fx-tc { font-size: 11px; }
.r-anchor .meta .desc {
  font-size: 13.5px; line-height: 1.55; color: var(--t2);
  text-wrap: pretty; max-width: 56ch;
}
.r-anchor .meta .tags { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 4px; }
.r-anchor .meta .actions {
  display: flex; align-items: center; gap: 8px; margin-top: 6px;
}

/* CAPTION */
.r-cp .caption {
  padding: 16px 24px 10px; display: flex; align-items: baseline; justify-content: space-between;
}
.r-cp .caption h2 {
  margin: 0; font-size: 18px; font-weight: 600; color: var(--t);
  letter-spacing: -0.012em;
}
.r-cp .caption h2 b { color: var(--ac); font-weight: 700; }
.r-cp .caption .meta {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  display: flex; align-items: center; gap: 12px;
}
.r-cp .caption .meta b { color: var(--t); font-weight: 500; }

/* GRID */
.r-grid {
  padding: 4px 24px 26px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
  align-content: start;
}

.r-echo {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 8px; overflow: hidden; cursor: pointer;
  display: flex; flex-direction: column; gap: 0;
  transition: border-color .12s, transform .12s, box-shadow .12s;
  position: relative;
}
.r-echo:hover { border-color: var(--bd2); transform: translateY(-1px); }
.r-echo.sel { border-color: var(--ac); box-shadow: 0 0 0 2px var(--ac-bg); }
.r-echo .kf {
  width: 100%; aspect-ratio: 4/3;
  background-size: cover; background-position: center;
  background-color: var(--bg);
  filter: contrast(1.04) brightness(0.96);
  position: relative;
}
.r-echo .kf .rank {
  position: absolute; top: 8px; left: 8px;
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  padding: 2px 7px; border-radius: 4px;
  background: rgba(14,16,20,0.78); color: var(--t);
}
.r-echo .kf .sim {
  position: absolute; top: 8px; right: 8px;
  font-family: var(--mono); font-size: 11px; font-weight: 700;
  padding: 2px 7px; border-radius: 4px;
  background: rgba(14,16,20,0.86); color: var(--ac);
  font-variant-numeric: tabular-nums;
}
.r-echo .kf .filmbadge {
  position: absolute; bottom: 8px; left: 8px;
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 8px; border-radius: 4px;
  background: rgba(14,16,20,0.86); color: var(--t);
  font-size: 11px; font-weight: 500;
}
.r-echo .kf .filmbadge .dot { width: 7px; height: 7px; border-radius: 50%; }
.r-echo .kf .tc {
  position: absolute; bottom: 8px; right: 8px;
  font-family: var(--mono); font-size: 10px;
  padding: 2px 6px; border-radius: 3px;
  background: rgba(14,16,20,0.78); color: var(--t);
}
.r-echo .body {
  padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 5px;
}
.r-echo .reason {
  font-size: 11.5px; color: var(--t2); line-height: 1.45;
  font-style: italic;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.r-echo .reason::before { content: '"'; color: var(--ac); margin-right: 2px; }
.r-echo .reason::after { content: '"'; color: var(--ac); margin-left: 2px; }
.r-echo .footrow {
  display: flex; align-items: center; justify-content: space-between; margin-top: 4px;
  font-size: 11px; color: var(--muted);
  font-family: var(--mono);
}
.r-echo .footrow .info { display: flex; align-items: center; gap: 7px; }

/* RIGHT PANE — Why this matches */
.r-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.r-rp .head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 13px 16px; border-bottom: 1px solid var(--bd);
}
.r-rp .head .l {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; color: var(--t);
}
.r-rp .head .l .pip {
  font-family: var(--mono); font-size: 10.5px;
  background: var(--ac); color: #fff; padding: 1px 6px; border-radius: 9px;
  font-weight: 600;
}
.r-rp .head .r { color: var(--muted); }

.r-rp .inner { padding: 16px 18px 18px; overflow-y: auto; flex: 1; }

/* Pair comparison */
.r-pair {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  margin-bottom: 16px;
}
.r-pair .cell {
  display: flex; flex-direction: column; gap: 6px;
}
.r-pair .cell .kf {
  width: 100%; aspect-ratio: 4/3; border-radius: 5px;
  background-size: cover; background-position: center;
  background-color: var(--bg);
  border: 1px solid var(--bd);
  filter: contrast(1.04) brightness(0.96);
  position: relative;
}
.r-pair .cell .lab {
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.r-pair .cell .lab.ac { color: var(--ac); }
.r-pair .cell .nm {
  display: flex; align-items: center; gap: 6px;
  font-size: 12.5px; color: var(--t); font-weight: 500;
}
.r-pair .cell .nm .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.r-pair .cell .nm .yr { font-family: var(--mono); font-size: 10px; color: var(--muted); }
.r-pair .vs {
  position: absolute; left: 50%; top: 35%;
  transform: translate(-50%, -50%);
  font-family: var(--mono); font-size: 11px; color: var(--ac);
  background: var(--ac-bg); padding: 4px 8px; border-radius: 12px;
  border: 1px solid var(--ac-dim);
  letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
  z-index: 2;
}
.r-pair-wrap { position: relative; }

.r-similarity-card {
  padding: 12px 14px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
  margin-bottom: 16px;
}
.r-similarity-card .h {
  display: flex; align-items: baseline; justify-content: space-between;
  font-size: 11px; color: var(--muted); font-weight: 500;
  margin-bottom: 10px;
}
.r-similarity-card .h .v {
  font-family: var(--mono); font-size: 16px; color: var(--ac);
  font-weight: 700; font-variant-numeric: tabular-nums;
}
.r-similarity-card .row {
  display: grid; grid-template-columns: 92px 1fr 44px;
  align-items: center; gap: 10px; margin-bottom: 7px;
  font-family: var(--mono); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.r-similarity-card .row .lab {
  color: var(--muted); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;
}
.r-similarity-card .row .track {
  height: 5px; background: var(--bd); border-radius: 3px; position: relative;
}
.r-similarity-card .row .track::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac-dim); border-radius: 3px;
}
.r-similarity-card .row .v { text-align: right; color: var(--t); }
.r-similarity-card .row.fused .track::before { background: var(--ac); }
.r-similarity-card .row.fused .lab { color: var(--ac); }

.r-sect {
  display: flex; align-items: baseline; justify-content: space-between;
  margin: 18px 0 8px;
  font-size: 11px; color: var(--muted); font-weight: 500;
  letter-spacing: 0.04em;
}
.r-sect a { color: var(--ac); font-family: var(--mono); cursor: pointer; font-size: 10.5px; }

.r-shared-tags { display: flex; flex-wrap: wrap; gap: 5px; }

.r-reason {
  padding: 12px 14px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
  font-size: 12.5px; line-height: 1.55; color: var(--t2);
  text-wrap: pretty; font-style: italic;
}
.r-reason::before { content: '"'; color: var(--ac); margin-right: 3px; font-style: normal; }
.r-reason::after { content: '"'; color: var(--ac); margin-left: 3px; font-style: normal; }

.r-actions { display: flex; flex-direction: column; gap: 6px; margin-top: 16px; }
.r-actions .fx-btn { justify-content: space-between; }
.r-actions .fx-btn .kbd { font-family: var(--mono); }
`;

function ScreenRimas({ selected, setSelected }) {
  const films = window.FILMS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  // Anchor scene: a strong rural exterior frame from Jeca Tatu
  const anchor = {
    film: 'jeca',
    kf: 'keyframes/kf-03-horse.jpg',
    tc: '00:01:57:18',
    cena: 3,
    desc: 'A rider on a pale horse pauses at the field\u2019s edge with figures in middle distance, mountain horizon, soft late-morning haze.',
    tags: ['exterior','horse-rider','rural-field','dia','sertão']
  };
  const anchorFilm = byId[anchor.film];

  // 8 echoes from OTHER films, ordered by similarity
  const echoes = [
    { film:'cangaceiro', kf:'keyframes/kf-05-man-cow.jpg',    tc:'00:42:11:07', cena:217, sim:0.94, reason:'figure + livestock, mid-frame, dawn light, ground tilted right' },
    { film:'aruanda',    kf:'keyframes/kf-07-woman-pot.jpg',  tc:'00:08:22:13', cena:34,  sim:0.91, reason:'solitary figure framed by thatched roof, horizon line at lower third' },
    { film:'pagador',    kf:'keyframes/kf-06-women-hut.jpg',  tc:'00:14:42:00', cena:158, sim:0.88, reason:'two figures at threshold of shelter, oblique camera, mid-grey gradient' },
    { film:'rio40',      kf:'keyframes/kf-04-cow.jpg',        tc:'00:54:03:18', cena:182, sim:0.86, reason:'rural still life, cloudy sky pressure, wagon as anchor object' },
    { film:'cangaceiro', kf:'keyframes/kf-11-mustache.jpg',   tc:'01:08:22:14', cena:402, sim:0.83, reason:'two-shot in open field, dry grass texture, shallow grading' },
    { film:'limite',     kf:'keyframes/kf-08-woman-dark.jpg', tc:'00:48:11:07', cena:84,  sim:0.79, reason:'interior counter-rhyme: figure with horizon-line equivalent indoors' },
    { film:'pagador',    kf:'keyframes/kf-02-fence.jpg',      tc:'00:22:09:11', cena:46,  sim:0.77, reason:'fence as horizontal anchor, figure pushed to upper third, rural scrub' },
    { film:'aruanda',    kf:'keyframes/kf-12-mustache2.jpg',  tc:'00:03:08:06', cena:11,  sim:0.75, reason:'duo at field’s edge, sketchy haze, similar focal length' },
  ];

  const sel = echoes[selected % echoes.length];
  const selFilm = byId[sel.film];

  // Mock signals derived from sim
  const sigVis = sel.sim;
  const sigSem = Math.min(0.95, sel.sim - 0.04);
  const sigCol = Math.min(0.93, sel.sim - 0.10);
  const sigComp = Math.min(0.96, sel.sim + 0.02);

  // Shared tags between anchor and selected echo (mock)
  const sharedTags = ['exterior', 'dia', 'rural-field'].filter(t => Math.random() > 0); // always

  return (
    <>
      <section className="r-cp">
        <div className="topbar">
          <div className="l">
            <div className="h">
              <span className="ic"><I.rhymes /></span>
              <span>Rimas visuais</span>
            </div>
            <span className="ct"><b>8</b> rimas em 5 filmes</span>
            <span style={{fontSize:12, color: FX.muted}}>· âncora · <b style={{color: FX.t}}>{anchorFilm.title}</b> cena {String(anchor.cena).padStart(3,'0')}</span>
          </div>
          <div className="r">
            <button className="fx-btn secondary"><I.image /> Trocar âncora</button>
            <button className="fx-btn secondary"><I.share /> Salvar coleção</button>
          </div>
        </div>

        <div className="controls">
          <span className="knob"><span className="k">Modelo</span><span className="v">CLIP-L / 14</span></span>
          <span className="knob"><span className="k">Distância</span><span className="v">cosine</span></span>
          <span className="knob"><span className="k">MMR</span><span className="v">λ 0.5</span></span>
          <span className="knob"><span className="k">Cross-film</span><span className="v acc">obrigatório</span></span>
          <span className="knob"><span className="k">k</span><span className="v">8</span></span>
          <span className="knob"><span className="k">Limiar</span><span className="v">0.75</span></span>
          <span className="grow"></span>
          <div className="modes">
            <span className="m on">Galeria</span>
            <span className="m">Pares</span>
            <span className="m">Constelação</span>
          </div>
          <button className="fx-icbtn"><I.sort /></button>
          <button className="fx-icbtn"><I.filter /></button>
          <button className="fx-icbtn"><I.more /></button>
        </div>

        <div className="scroll">
          {/* ANCHOR SECTION */}
          <div className="r-anchor">
            <div className="kf-wrap">
              <div className="kf" style={{backgroundImage:`url(${anchor.kf})`}}></div>
              <span className="kf-label">
                <span className="gly"></span>
                Âncora
              </span>
            </div>
            <div className="meta">
              <div className="ll">
                <span className="dot"></span>
                <span>Cena de referência · embedding fixo</span>
              </div>
              <h1>Cavaleiro à beira do campo</h1>
              <div className="filmrow">
                <span className="pill">
                  <span className="dot" style={{background: FX_FILM[anchor.film]}}></span>
                  <b>{anchorFilm.title}</b>
                  <span style={{fontFamily:'var(--mono)', fontSize: 11, color: FX.muted}}>{anchorFilm.year}</span>
                </span>
                <span style={{color: FX.muted, fontSize: 12}}>dir. {anchorFilm.director}</span>
              </div>
              <div className="ids">
                <span>cena {String(anchor.cena).padStart(3,'0')} / {anchorFilm.scenes}</span>
                <span>·</span>
                <span className="fx-tc bare">{anchor.tc}</span>
                <span>·</span>
                <span>~ 4.2s</span>
              </div>
              <p className="desc">{anchor.desc}</p>
              <div className="tags">
                {anchor.tags.map((t,i) => (
                  <span key={i} className={'fx-pill' + (i < 3 ? ' ac' : '')}>{t}</span>
                ))}
              </div>
              <div className="actions">
                <button className="fx-btn secondary"><I.play /> Abrir cena</button>
                <button className="fx-btn secondary"><I.comment /> Comentários (2)</button>
                <button className="fx-btn ghost"><I.more /></button>
              </div>
            </div>
          </div>

          {/* CAPTION */}
          <div className="caption">
            <h2><b>8</b> rimas em 5 filmes</h2>
            <span className="meta">
              <span>k=8 · MMR λ 0.5</span>
              <span>·</span>
              <b>187 ms</b>
              <span>·</span>
              <span>CLIP-L/14 · cosine</span>
            </span>
          </div>

          {/* GRID */}
          <div className="r-grid">
            {echoes.map((e, i) => {
              const ff = byId[e.film];
              return (
                <article key={i}
                         className={'r-echo' + ((i === selected % echoes.length) ? ' sel' : '')}
                         onClick={() => setSelected(i)}>
                  <div className="kf" style={{backgroundImage:`url(${e.kf})`}}>
                    <span className="rank">#{String(i+1).padStart(2,'0')}</span>
                    <span className="sim">{e.sim.toFixed(2)}</span>
                    <span className="filmbadge">
                      <span className="dot" style={{background: FX_FILM[e.film]}}></span>
                      {ff.title}
                    </span>
                    <span className="tc">{e.tc.slice(0,8)}</span>
                  </div>
                  <div className="body">
                    <p className="reason">{e.reason}</p>
                    <div className="footrow">
                      <span className="info">{ff.year} · cena {String(e.cena).padStart(3,'0')}</span>
                      <span style={{color: FX.ac}}>↗ abrir</span>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      {/* RIGHT — Why this matches */}
      <aside className="r-rp">
        <div className="head">
          <div className="l">
            <span className="pip">#{(selected % echoes.length) + 1}</span>
            <span>Por que esta rima</span>
          </div>
          <div className="r">
            <button className="fx-icbtn sm"><I.more /></button>
          </div>
        </div>

        <div className="inner">
          {/* Pair comparison */}
          <div className="r-pair-wrap">
            <div className="r-pair">
              <div className="cell">
                <span className="lab ac">Âncora</span>
                <div className="kf" style={{backgroundImage:`url(${anchor.kf})`}}></div>
                <span className="nm">
                  <span className="dot" style={{background: FX_FILM[anchor.film]}}></span>
                  {anchorFilm.title}
                  <span className="yr">{anchorFilm.year}</span>
                </span>
                <span style={{fontFamily:'var(--mono)', fontSize: 10.5, color: FX.muted}}>cena {String(anchor.cena).padStart(3,'0')} · {anchor.tc}</span>
              </div>
              <div className="cell">
                <span className="lab">Rima · #{(selected % echoes.length)+1}</span>
                <div className="kf" style={{backgroundImage:`url(${sel.kf})`}}></div>
                <span className="nm">
                  <span className="dot" style={{background: FX_FILM[sel.film]}}></span>
                  {selFilm.title}
                  <span className="yr">{selFilm.year}</span>
                </span>
                <span style={{fontFamily:'var(--mono)', fontSize: 10.5, color: FX.muted}}>cena {String(sel.cena).padStart(3,'0')} · {sel.tc}</span>
              </div>
            </div>
          </div>

          {/* Reason */}
          <div className="r-sect">
            <span>Reasoning · md2-explain</span>
            <a>re-gerar</a>
          </div>
          <p className="r-reason">{sel.reason}</p>

          {/* Similarity card */}
          <div className="r-sect" style={{marginTop: 22}}>
            <span>Sinais de similaridade</span>
            <a>detalhes</a>
          </div>
          <div className="r-similarity-card">
            <div className="h">
              <span>cosine · embedding fundido</span>
              <span className="v">{sel.sim.toFixed(3)}</span>
            </div>
            <div className="row"><span className="lab">Visual · CLIP</span><span className="track" style={{'--p': `${(sigVis*100).toFixed(0)}%`}}></span><span className="v">{sigVis.toFixed(3)}</span></div>
            <div className="row"><span className="lab">Composição</span><span className="track" style={{'--p': `${(sigComp*100).toFixed(0)}%`}}></span><span className="v">{sigComp.toFixed(3)}</span></div>
            <div className="row"><span className="lab">Semântico</span><span className="track" style={{'--p': `${(sigSem*100).toFixed(0)}%`}}></span><span className="v">{sigSem.toFixed(3)}</span></div>
            <div className="row"><span className="lab">Cor / luma</span><span className="track" style={{'--p': `${(sigCol*100).toFixed(0)}%`}}></span><span className="v">{sigCol.toFixed(3)}</span></div>
            <div className="row fused"><span className="lab">Fundido</span><span className="track" style={{'--p': `${(sel.sim*100).toFixed(0)}%`}}></span><span className="v">{sel.sim.toFixed(3)}</span></div>
          </div>

          {/* Shared tags */}
          <div className="r-sect">
            <span>Tags compartilhadas · {sharedTags.length}</span>
            <a>todas</a>
          </div>
          <div className="r-shared-tags">
            {anchor.tags.slice(0,3).map((t, i) => (
              <span key={i} className="fx-pill ac">{t}</span>
            ))}
            <span className="fx-pill outline" style={{color: FX.muted, borderStyle:'dashed'}}>+ {anchor.tags.length - 3}</span>
          </div>

          {/* Actions */}
          <div className="r-actions">
            <button className="fx-btn primary">
              <span>Abrir cena de rima</span>
              <span className="kbd" style={{background:'rgba(0,0,0,0.2)', padding:'0 5px', borderRadius:3}}>⏎</span>
            </button>
            <button className="fx-btn secondary">
              <span><I.pin /> Salvar par à coleção</span>
              <span className="kbd">⌥S</span>
            </button>
            <button className="fx-btn secondary">
              <span><I.comment /> Anotar rima</span>
              <span className="kbd">A</span>
            </button>
            <button className="fx-btn secondary">
              <span><I.share /> Compartilhar</span>
              <span className="kbd">⌘C</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

window.ScreenRimas = ScreenRimas;
window.RIMAS_CSS = RIMAS_CSS;
