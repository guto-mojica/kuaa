// Mojica · Frame.io branch · Buscar screen (semantic search)

const BUSCAR_CSS = `
.b-screen { display: contents; }

.b-cp { display: flex; flex-direction: column; min-width: 0; overflow: hidden; background: var(--bg); }

/* Search row */
.b-cp .search-wrap { padding: 16px 24px 12px; border-bottom: 1px solid var(--bd); }
.b-cp .qbar { display: flex; align-items: center; gap: 10px; }
.b-cp .qbar .ico { color: var(--muted); font-size: 16px; }
.b-cp .qbar .qinput {
  flex: 1; background: var(--panel); border: 1px solid var(--bd);
  border-radius: 7px; padding: 9px 12px;
  display: flex; align-items: center; gap: 10px;
  transition: border-color .12s, background .12s, box-shadow .12s;
}
.b-cp .qbar .qinput:focus-within {
  border-color: var(--ac); background: var(--raised);
  box-shadow: 0 0 0 3px var(--ac-bg-low);
}
.b-cp .qbar .qinput input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; font-size: 14px; color: var(--t);
}
.b-cp .qbar .qinput input::placeholder { color: var(--muted); }
.b-cp .qbar .qinput .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 5px;
  border: 1px solid var(--bd2); border-radius: 3px; color: var(--muted);
}

.b-cp .modes {
  display: flex; align-items: center; gap: 8px; padding-top: 12px;
  flex-wrap: wrap;
}
.b-cp .chip {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 6px;
  background: var(--panel); border: 1px solid var(--bd);
  color: var(--t2); font-size: 12px; cursor: pointer; font-weight: 500;
}
.b-cp .chip:hover { background: var(--hover); border-color: var(--bd2); color: var(--t); }
.b-cp .chip.on { background: var(--ac-bg); border-color: var(--ac-dim); color: var(--ac); }
.b-cp .chip.on .ico { color: var(--ac); }
.b-cp .chip .ico { color: var(--muted); font-size: 13px; display: flex; align-items: center; }
.b-cp .div-v { width: 1px; height: 22px; background: var(--bd); margin: 0 6px; }
.b-cp .knob {
  display: flex; align-items: center; gap: 7px;
  font-size: 11.5px; color: var(--muted);
}
.b-cp .knob .k {
  text-transform: uppercase; letter-spacing: 0.06em;
  font-size: 10.5px; color: var(--faint); font-family: var(--mono);
}
.b-cp .knob .v {
  color: var(--t2); font-family: var(--mono); font-size: 11px;
  padding: 2px 7px; background: var(--panel); border: 1px solid var(--bd);
  border-radius: 4px; font-variant-numeric: tabular-nums;
}
.b-cp .knob .v.acc { color: var(--ac); border-color: var(--ac-dim); }

/* Caption */
.b-cp .caption {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 24px; border-bottom: 1px solid var(--bd); background: var(--bg);
}
.b-cp .caption .left {
  display: flex; align-items: center; gap: 14px; font-size: 13px;
}
.b-cp .caption .left .ttl { color: var(--t); font-weight: 600; }
.b-cp .caption .left .ttl b { color: var(--ac); font-weight: 700; }
.b-cp .caption .left .meta {
  display: flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}
.b-cp .caption .left .meta b { color: var(--t); font-weight: 500; }
.b-cp .caption .right { display: flex; align-items: center; gap: 6px; }
.b-cp .caption .right .seg {
  font-size: 11.5px; color: var(--muted); padding: 4px 9px; border-radius: 5px;
  cursor: pointer; font-weight: 500; display: flex; align-items: center; gap: 6px;
}
.b-cp .caption .right .seg:hover { background: var(--hover); color: var(--t); }
.b-cp .caption .right .seg.on { background: var(--hover); color: var(--t); }

/* GRID */
.b-cp .grid {
  flex: 1; overflow-y: auto;
  padding: 18px 24px 20px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(282px, 1fr));
  gap: 16px;
  align-content: start;
}
.b-card {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 8px; overflow: hidden;
  cursor: pointer; display: flex; flex-direction: column;
  transition: border-color .12s, transform .12s;
}
.b-card:hover { border-color: var(--bd2); transform: translateY(-1px); }
.b-card.sel { border-color: var(--ac); box-shadow: 0 0 0 2px var(--ac-bg); }
.b-card .kf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--bg) center/cover no-repeat;
  position: relative; filter: contrast(1.04) brightness(0.96);
}
.b-card .kf .tl {
  position: absolute; top: 8px; left: 8px;
  display: flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(14,16,20,0.78); border-radius: 4px;
  backdrop-filter: blur(4px); letter-spacing: 0.02em;
}
.b-card .kf .tl .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
.b-card .kf .bl {
  position: absolute; bottom: 8px; left: 8px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(14,16,20,0.78); border-radius: 4px;
  font-variant-numeric: tabular-nums;
}
.b-card .kf .tr {
  position: absolute; top: 8px; right: 8px;
  font-family: var(--mono); font-size: 11px; color: var(--ac);
  padding: 3px 7px; background: rgba(14,16,20,0.78); border-radius: 4px;
  font-weight: 600; backdrop-filter: blur(4px);
}
.b-card .kf .br {
  position: absolute; bottom: 8px; right: 8px;
  display: flex; align-items: center; gap: 4px;
  font-family: var(--mono); font-size: 10px; color: var(--yellow);
  padding: 3px 7px; background: rgba(14,16,20,0.78); border-radius: 4px;
}
.b-card .kf .br .pin { width: 6px; height: 6px; border-radius: 50%; background: var(--yellow); }

.b-card .body { padding: 11px 12px 12px; display: flex; flex-direction: column; gap: 6px; }
.b-card .head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
}
.b-card .filmname {
  display: flex; align-items: center; gap: 7px;
  font-size: 13px; font-weight: 600; color: var(--t);
  min-width: 0;
}
.b-card .filmname .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.b-card .filmname .nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.b-card .yr { font-family: var(--mono); font-size: 10.5px; color: var(--muted); flex-shrink: 0; }

.b-card .desc {
  font-size: 12.5px; line-height: 1.5; color: var(--t2);
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}
.b-card .footrow {
  display: flex; align-items: center; justify-content: space-between;
  padding-top: 8px; border-top: 1px solid var(--bd);
}
.b-card .footrow .tags { display: flex; gap: 4px; flex-wrap: wrap; }
.b-card .footrow .acts { display: flex; gap: 2px; color: var(--muted); }

/* RIGHT PANE */
.b-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.b-rp .htabs {
  display: flex; align-items: center; padding: 0 12px;
  border-bottom: 1px solid var(--bd);
  gap: 2px;
}
.b-rp .htab {
  padding: 13px 12px; font-size: 12.5px; color: var(--muted);
  cursor: pointer; position: relative; font-weight: 500;
  display: flex; align-items: center; gap: 6px;
}
.b-rp .htab:hover { color: var(--t); }
.b-rp .htab.on { color: var(--t); }
.b-rp .htab.on::after {
  content: ''; position: absolute; left: 8px; right: 8px; bottom: -1px;
  height: 2px; background: var(--ac); border-radius: 1px;
}
.b-rp .htab .pip {
  font-family: var(--mono); font-size: 10px; padding: 0 5px;
  background: var(--raised); border-radius: 8px; color: var(--t2);
}
.b-rp .htabs .gap { flex: 1; }
.b-rp .htabs .ic { color: var(--muted); cursor: pointer; padding: 6px; border-radius: 4px; }
.b-rp .htabs .ic:hover { background: var(--hover); color: var(--t); }

.b-rp .inner { padding: 14px 16px 18px; overflow-y: auto; flex: 1; }

/* big insp kf with pin */
.b-rp .insp-kf {
  width: 100%; aspect-ratio: 16/10; border-radius: 6px;
  position: relative; overflow: hidden;
  background: var(--bg) center/cover no-repeat;
  filter: contrast(1.05) brightness(0.97);
  border: 1px solid var(--bd);
}
.b-rp .insp-kf .pin {
  position: absolute; top: 28%; left: 18%;
  width: 22px; height: 22px; border-radius: 11px 11px 11px 0;
  background: var(--yellow); color: #0E1014;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 800; font-family: var(--mono);
  transform: rotate(-12deg);
  box-shadow: 0 0 0 3px rgba(245,200,66,0.25);
  cursor: pointer;
}
.b-rp .insp-kf .annx {
  position: absolute; bottom: 10px; left: 10px;
  display: flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(14,16,20,0.86); border-radius: 4px;
}
.b-rp .insp-kf .annx .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--yellow); }

.b-rp .meta-top {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 10px; margin-top: 14px;
}
.b-rp .meta-top h3 {
  margin: 0; font-size: 16px; font-weight: 600; color: var(--t);
  letter-spacing: -0.01em; line-height: 1.25;
}
.b-rp .meta-top .at {
  display: flex; align-items: center; gap: 7px; margin-top: 5px;
  font-size: 11.5px; color: var(--muted);
}
.b-rp .meta-top .at .dot { width: 7px; height: 7px; border-radius: 50%; }
.b-rp .meta-top .at b { color: var(--t2); font-weight: 500; }

/* thread */
.b-thread { margin-top: 18px; display: flex; flex-direction: column; gap: 14px; }
.b-com {
  display: grid; grid-template-columns: 30px 1fr; gap: 10px;
}
.b-com .av {
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--raised); display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10.5px; font-weight: 600; color: var(--t);
}
.b-com.ai .av { background: linear-gradient(135deg, var(--ac), var(--ac-dim)); color: #fff; }
.b-com.curator .av { background: linear-gradient(135deg, var(--pink), #B0432B); color: #fff; }
.b-com .bx { display: flex; flex-direction: column; gap: 5px; }
.b-com .who {
  display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap;
  font-size: 12px;
}
.b-com .who .n { color: var(--t); font-weight: 500; }
.b-com .who .badge {
  font-family: var(--mono); font-size: 9.5px; padding: 1px 5px;
  border-radius: 3px; background: var(--raised); color: var(--t2);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.b-com.ai .who .badge { background: var(--ac-bg); color: var(--ac); }
.b-com .who .when {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
.b-com .body {
  font-size: 12.5px; line-height: 1.55; color: var(--t2);
  text-wrap: pretty;
}
.b-com .body .fx-tc { margin-right: 4px; }
.b-com .replyrow {
  display: flex; align-items: center; gap: 12px;
  font-size: 11px; color: var(--muted);
  margin-top: 2px;
}
.b-com .replyrow a { color: var(--muted); cursor: pointer; }
.b-com .replyrow a:hover { color: var(--ac); }
.b-com.pinned .who .badge { background: var(--yellow-bg); color: var(--yellow); }

/* signals card */
.b-sigs {
  margin-top: 18px; padding: 14px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 7px;
}
.b-sigs .h {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 12px;
  font-size: 11px; color: var(--muted); font-weight: 500;
}
.b-sigs .h .v {
  font-family: var(--mono); color: var(--t); font-variant-numeric: tabular-nums;
}
.b-sigs .row {
  display: grid; grid-template-columns: 76px 1fr 44px;
  align-items: center; gap: 10px; margin-bottom: 7px;
  font-family: var(--mono); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.b-sigs .row .lab {
  color: var(--muted); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;
}
.b-sigs .row .track {
  height: 5px; background: var(--bd); border-radius: 3px; position: relative;
}
.b-sigs .row .track::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac-dim); border-radius: 3px;
}
.b-sigs .row .v { text-align: right; color: var(--t); }
.b-sigs .row.fused .track::before { background: var(--ac); }
.b-sigs .row.fused .lab { color: var(--ac); }

/* rimas */
.b-rimas { margin-top: 16px; padding: 12px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 7px;
}
.b-rimas .h {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 10px; font-size: 11px; color: var(--muted); font-weight: 500;
}
.b-rimas .h a {
  color: var(--ac); cursor: pointer; font-size: 10.5px;
  display: flex; align-items: center; gap: 4px;
}
.b-rimas .gr3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.b-rimas .ry { cursor: pointer; }
.b-rimas .ry .kf {
  width: 100%; aspect-ratio: 4/3;
  background-size: cover; background-position: center;
  border-radius: 4px; filter: contrast(1.05) brightness(0.94);
  border: 1px solid var(--bd); transition: border-color .12s;
}
.b-rimas .ry:hover .kf { border-color: var(--ac); }
.b-rimas .ry .lab {
  font-size: 10.5px; color: var(--t2); margin-top: 4px;
  display: flex; align-items: center; gap: 5px;
}
.b-rimas .ry .lab .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.b-rimas .ry .lab .nm { font-weight: 500; }
.b-rimas .ry .lab .sc { color: var(--ac); font-family: var(--mono); margin-left: auto; }

/* composer */
.b-comp {
  margin-top: 16px; padding: 10px 12px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 7px;
}
.b-comp textarea {
  width: 100%; resize: none; min-height: 44px;
  background: transparent; border: none; outline: none;
  font: inherit; font-size: 12.5px; color: var(--t);
}
.b-comp textarea::placeholder { color: var(--muted); }
.b-comp .actrow {
  display: flex; align-items: center; justify-content: space-between; margin-top: 6px;
}
.b-comp .actrow .tools { display: flex; gap: 1px; color: var(--muted); }
.b-comp .actrow .tools .tool {
  width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
  border-radius: 4px; cursor: pointer;
}
.b-comp .actrow .tools .tool:hover { background: var(--hover); color: var(--t); }
.b-comp .actrow .left {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
.b-comp .actrow .left .fx-tc { font-weight: 600; }

/* bottom timeline strip */
.b-tl {
  border-top: 1px solid var(--bd); background: var(--panel);
  padding: 10px 24px 12px;
  display: flex; flex-direction: column; gap: 8px;
}
.b-tl .head {
  display: flex; align-items: center; justify-content: space-between;
}
.b-tl .ttitle {
  display: flex; align-items: center; gap: 10px;
  font-size: 12px; color: var(--t); font-weight: 500;
}
.b-tl .ttitle .dot { width: 8px; height: 8px; border-radius: 50%; }
.b-tl .ctrls {
  display: flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}
.b-tl .ctrls .tc { color: var(--yellow); font-variant-numeric: tabular-nums; font-weight: 600; }
.b-tl .ctrls .ib {
  width: 26px; height: 24px; display: flex; align-items: center; justify-content: center;
  border-radius: 4px; background: var(--raised); color: var(--t2); cursor: pointer;
}
.b-tl .ctrls .ib:hover { background: var(--hover); color: var(--t); }
.b-tl .scrub {
  position: relative; height: 50px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 5px;
  display: flex; gap: 0; overflow: hidden;
}
.b-tl .seg {
  flex: 1; min-width: 0; height: 100%;
  background-size: cover; background-position: center;
  border-right: 1px solid rgba(0,0,0,0.5);
  filter: brightness(0.55) contrast(1.05);
  position: relative; cursor: pointer; transition: filter .15s;
}
.b-tl .seg:hover { filter: brightness(0.9) contrast(1.05); }
.b-tl .seg.match::after {
  content: ''; position: absolute; left: 0; right: 0; bottom: 0;
  height: 3px; background: var(--ac);
}
.b-tl .seg.sel { filter: brightness(1.0) contrast(1.1); }
.b-tl .seg.sel::before {
  content: ''; position: absolute; inset: 0;
  outline: 2px solid var(--yellow); outline-offset: -2px; z-index: 2;
}
.b-tl .seg.sel::after { background: var(--yellow); height: 4px; }
.b-tl .ticks {
  display: grid; grid-template-columns: repeat(8, 1fr);
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
}
`;

function ScreenBuscar({ selected, setSelected }) {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  const r = results[selected];
  const f = byId[r.film];

  const sigSem  = Math.min(0.96, r.score + 0.06);
  const sigBm25 = Math.max(0.04, r.score - 0.52);
  const sigRk   = Math.min(0.96, r.score - 0.01);
  const sigFu   = r.score;

  const rhymes = results.filter((x, i) => x.film !== r.film && i !== selected).slice(0, 3)
    .map((x, i) => ({...x, sim: (0.94 - i*0.04).toFixed(2)}));

  // timeline mock for current film
  const allKfs = [
    'keyframes/kf-01-title.jpg', 'keyframes/kf-02-fence.jpg', 'keyframes/kf-03-horse.jpg',
    'keyframes/kf-04-cow.jpg',   'keyframes/kf-05-man-cow.jpg', 'keyframes/kf-06-women-hut.jpg',
    'keyframes/kf-07-woman-pot.jpg', 'keyframes/kf-08-woman-dark.jpg', 'keyframes/kf-09-bed.jpg',
    'keyframes/kf-10-shirt.jpg', 'keyframes/kf-11-mustache.jpg', 'keyframes/kf-12-mustache2.jpg',
    'keyframes/kf-13-conversation.jpg', 'keyframes/kf-14-brinquinho.jpg',
    'keyframes/kf-15-night-fence.jpg', 'keyframes/kf-16-flames.jpg',
    'keyframes/kf-17-smoke.jpg', 'keyframes/kf-18-night-fire.jpg',
  ];
  const tl = Array.from({length: 24}, (_, i) => allKfs[i % allKfs.length]);
  const matchedIdx = [3, 7, 11, 16, 21];

  const filmMatchCounts = {};
  results.forEach(rr => filmMatchCounts[rr.film] = (filmMatchCounts[rr.film] || 0) + 1);

  return (
    <>
      <section className="b-cp">
        <div className="search-wrap">
          <div className="qbar">
            <div className="qinput">
              <span style={{color: FX.muted, display:'flex'}}><I.search /></span>
              <input defaultValue="duas pessoas conversando ao ar livre" />
              <span className="kbd">⌘K</span>
            </div>
            <button className="fx-btn primary"
                    data-tip="Re-executar busca · ⏎"
                    onClick={() => window.ToastBus && window.ToastBus.push({
                      kind: 'success',
                      title: '9 cenas em 6 filmes',
                      sub: <>sem 0.70 · bm25 0.30 · rerank <span style={{color:'var(--ac)'}}>on</span> · <span style={{fontFamily:'var(--mono)',color:'var(--ac)'}}>231 ms</span></>,
                    })}>
              Buscar <span className="kbd">⏎</span>
            </button>
          </div>
          <div className="modes">
            <span className="chip on"><span className="ico"><I.tag /></span>Texto</span>
            <span className="chip"><span className="ico"><I.image /></span>Imagem</span>
            <span className="chip"><span className="ico"><I.audio /></span>Trilha</span>
            <span className="chip"><span className="ico">⊕</span>Multimodal</span>
            <span className="div-v"></span>
            <span className="knob"><span className="k">Híbrido</span><span className="v">sem 0.70 · bm25 0.30</span></span>
            <span className="knob"><span className="k">Rerank</span><span className="v acc">on</span></span>
            <span className="knob"><span className="k">MMR</span><span className="v">λ 0.50</span></span>
            <span className="knob"><span className="k">k</span><span className="v">9</span></span>
          </div>
        </div>

        <div className="caption">
          <div className="left">
            <span className="ttl"><b>9 cenas</b> em 6 filmes</span>
            <span className="meta">
              <span>·</span>
              <b>231 ms</b>
              <span>·</span>
              <span>sem ⊕ bm25 ⊕ rerank</span>
            </span>
          </div>
          <div className="right">
            <span className="seg on"><I.appearance /> Grade</span>
            <span className="seg"><I.sort /> Lista</span>
            <span className="seg"><I.group /> Compacto</span>
          </div>
        </div>

        <div className="grid">
          {results.map((rr, i) => {
            const ff = byId[rr.film];
            const hasAnn = (i === 0 || i === 3);
            return (
              <article key={rr.id}
                       className={'b-card' + (i === selected ? ' sel' : '')}
                       onClick={() => setSelected(i)}>
                <div className="kf" style={{backgroundImage:`url(${rr.kf})`}}>
                  <span className="tl"><span className="dot"></span>indexado</span>
                  <span className="bl">{rr.tc}</span>
                  <span className="tr">{rr.score.toFixed(3)}</span>
                  {hasAnn && <span className="br"><span className="pin"></span>1 pin</span>}
                </div>
                <div className="body">
                  <div className="head">
                    <span className="filmname">
                      <span className="dot" style={{background: FX_FILM[rr.film]}}></span>
                      <span className="nm">{ff.title}</span>
                    </span>
                    <span className="yr">{ff.year} · #{String(rr.cena).padStart(3,'0')}</span>
                  </div>
                  <p className="desc">{rr.desc}</p>
                  <div className="footrow">
                    <div className="tags">
                      {rr.tags.slice(0,3).map((t,j) => (
                        <span key={j} className={'fx-pill' + (t==='duas-pessoas'||t==='exterior' ? ' ac' : '')}>{t}</span>
                      ))}
                    </div>
                    <div className="acts">
                      <button className="fx-icbtn sm"><I.more /></button>
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>

        {/* TIMELINE for selected film */}
        <div className="b-tl">
          <div className="head">
            <div className="ttitle">
              <span className="dot" style={{background: FX_FILM[r.film]}}></span>
              <span>Timeline · {f.title}</span>
              <span className="fx-pill">{f.scenes} cenas</span>
              <span className="fx-pill ac">{filmMatchCounts[r.film]} matches</span>
            </div>
            <div className="ctrls">
              <span className="ib"><I.expand /></span>
              <span>00:00</span>
              <span className="fx-tc bare">{r.tc}</span>
              <span>{Math.floor(f.runtime/60)}:{String(f.runtime%60).padStart(2,'0')}:00</span>
              <span className="ib"><I.play /></span>
            </div>
          </div>
          <div className="scrub">
            {tl.map((kf, i) => {
              const isSel = i === 11;
              const isMatch = matchedIdx.includes(i);
              return (
                <span key={i}
                      className={'seg' + (isMatch ? ' match' : '') + (isSel ? ' sel' : '')}
                      style={{backgroundImage:`url(${kf})`}}></span>
              );
            })}
          </div>
          <div className="ticks">
            <span>00:00</span><span>12:00</span><span>24:00</span><span>36:00</span>
            <span>48:00</span><span>60:00</span><span>72:00</span><span>96:00</span>
          </div>
        </div>
      </section>

      {/* RIGHT — comments + signals */}
      <aside className="b-rp">
        <div className="htabs">
          <span className="htab on"><I.comment /> Atividade <span className="pip">4</span></span>
          <span className="htab"><I.pin /> Anotações <span className="pip">1</span></span>
          <span className="htab">Propriedades</span>
          <span className="gap"></span>
          <span className="ic"><I.more /></span>
        </div>
        <div className="inner">
          <div className="insp-kf" style={{backgroundImage:`url(${r.kf})`}}>
            <div className="pin">1</div>
            <span className="annx"><span className="dot"></span>1 anotação · {r.tc}</span>
          </div>

          <div className="meta-top">
            <div>
              <h3>Cena {String(r.cena).padStart(3,'0')} · {f.title}</h3>
              <div className="at">
                <span className="dot" style={{background: FX_FILM[r.film]}}></span>
                <b>{f.title}</b>
                <span>·</span>
                <span>{f.year}</span>
                <span>·</span>
                <span>{f.director}</span>
                <span>·</span>
                <span className="fx-tc bare">{r.tc}</span>
              </div>
            </div>
            <span className="fx-pill green">
              <span className="dot"></span>
              indexado
            </span>
          </div>

          {/* THREAD */}
          <div className="b-thread">
            <div className="b-com ai">
              <div className="av">md</div>
              <div className="bx">
                <div className="who">
                  <span className="n">moondream-2</span>
                  <span className="badge">AI · descrição</span>
                  <span className="when">há 4 dias</span>
                </div>
                <div className="body">
                  <span className="fx-tc">{r.tc}</span>
                  {r.desc}
                </div>
                <div className="replyrow">
                  <a>Responder</a>
                  <a>Editar</a>
                  <a>Re-gerar</a>
                </div>
              </div>
            </div>

            <div className="b-com curator pinned">
              <div className="av">RG</div>
              <div className="bx">
                <div className="who">
                  <span className="n">Rafael · curador</span>
                  <span className="badge">📍 pin · {r.tc}</span>
                  <span className="when">há 2h</span>
                </div>
                <div className="body">
                  Cena de referência para a vertente <b>"diálogos no campo aberto"</b>. Boa candidata para o corte da retrospectiva 2026.
                </div>
                <div className="replyrow">
                  <a>Responder</a>
                  <a>Resolver</a>
                </div>
              </div>
            </div>
          </div>

          <div className="b-sigs">
            <div className="h">
              <span>Por que este resultado</span>
              <span className="v">{r.score.toFixed(3)}</span>
            </div>
            <div className="row"><span className="lab">Semântico</span><span className="track" style={{'--p': `${(sigSem*100).toFixed(0)}%`}}></span><span className="v">{sigSem.toFixed(3)}</span></div>
            <div className="row"><span className="lab">BM25</span><span className="track" style={{'--p': `${(sigBm25*100).toFixed(0)}%`}}></span><span className="v">{sigBm25.toFixed(3)}</span></div>
            <div className="row"><span className="lab">Rerank</span><span className="track" style={{'--p': `${(sigRk*100).toFixed(0)}%`}}></span><span className="v">{sigRk.toFixed(3)}</span></div>
            <div className="row fused"><span className="lab">Fundido</span><span className="track" style={{'--p': `${(sigFu*100).toFixed(0)}%`}}></span><span className="v">{sigFu.toFixed(3)}</span></div>
          </div>

          <div className="b-rimas">
            <div className="h">
              <span>Rimas visuais · cross-film</span>
              <a>Ver todas <I.arrowR /></a>
            </div>
            <div className="gr3">
              {rhymes.map((x, i) => {
                const ff = byId[x.film];
                return (
                  <div key={i} className="ry">
                    <div className="kf" style={{backgroundImage:`url(${x.kf})`}}></div>
                    <div className="lab">
                      <span className="dot" style={{background: FX_FILM[x.film]}}></span>
                      <span className="nm">{ff.title}</span>
                      <span className="sc">{x.sim}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="b-comp">
            <textarea placeholder="Adicionar comentário ou anotação…"></textarea>
            <div className="actrow">
              <div className="left">
                <span className="fx-tc">{r.tc}</span>
              </div>
              <div className="tools">
                <span className="tool" title="Pin"><I.pin /></span>
                <span className="tool" title="Tag"><I.tag /></span>
                <span className="tool" title="Attach"><I.attach /></span>
                <span className="tool" title="Emoji"><I.emoji /></span>
                <button className="fx-btn primary" style={{padding:'5px 11px', marginLeft: 4}}>
                  <I.send /> Comentar
                </button>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

window.ScreenBuscar = ScreenBuscar;
window.BUSCAR_CSS = BUSCAR_CSS;
