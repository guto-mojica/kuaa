// Mojica · Frame.io branch · Processamento screen (active pipeline)
// 5-step pipeline: Frames → Cenas → Visual → Embeddings → Descrições
// SSE-style live log with the currently-processing film highlighted.

const PROC_CSS = `
.p-cp { display: flex; flex-direction: column; min-width: 0; overflow: hidden; background: var(--bg); }

.p-top {
  padding: 16px 24px; border-bottom: 1px solid var(--bd);
}
.p-top .row1 {
  display: flex; align-items: center; justify-content: space-between; gap: 18px;
}
.p-top h1 {
  margin: 0; display: flex; align-items: center; gap: 12px;
  font-size: 18px; font-weight: 600; color: var(--t); letter-spacing: -0.012em;
}
.p-top h1 .ic {
  width: 30px; height: 30px; border-radius: 7px;
  background: var(--orange-bg); color: var(--orange);
  display: flex; align-items: center; justify-content: center;
}
.p-top h1 .pip {
  font-family: var(--mono); font-size: 11.5px;
  background: var(--orange); color: #0E1014;
  padding: 2px 8px; border-radius: 9px; font-weight: 600;
  margin-left: 2px;
}
.p-top .acts { display: flex; align-items: center; gap: 8px; }

.p-active {
  margin-top: 16px;
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 8px; padding: 16px 18px;
  display: flex; flex-direction: column; gap: 14px;
  position: relative;
}
.p-active::before {
  content: ''; position: absolute; left: 0; top: 10px; bottom: 10px;
  width: 3px; background: var(--orange); border-radius: 2px;
}
.p-active .head {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
}
.p-active .head .l {
  display: flex; align-items: center; gap: 14px;
}
.p-active .head .thumb {
  width: 76px; height: 56px; border-radius: 5px;
  background-size: cover; background-position: center;
  border: 1px solid var(--bd2); filter: contrast(1.05) brightness(0.96);
  flex-shrink: 0;
}
.p-active .head .info { display: flex; flex-direction: column; gap: 3px; }
.p-active .head h2 {
  margin: 0; font-size: 16px; font-weight: 600; color: var(--t);
  letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 9px;
}
.p-active .head h2 .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--orange); }
.p-active .head h2 .stat {
  font-family: var(--mono); font-size: 10.5px; padding: 1px 7px;
  background: var(--orange-bg); color: var(--orange);
  border-radius: 9px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase;
}
.p-active .head .sub {
  display: flex; align-items: center; gap: 8px;
  font-size: 11.5px; color: var(--muted);
}
.p-active .head .sub b { color: var(--t2); font-weight: 500; }
.p-active .head .r {
  display: flex; align-items: center; gap: 6px;
}

/* Big progress */
.p-pbar {
  position: relative; height: 7px;
  background: var(--bd); border-radius: 4px;
  overflow: hidden;
}
.p-pbar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: linear-gradient(90deg, var(--ac), var(--orange));
  border-radius: 4px;
}
.p-pbar::after {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p);
  background-image: repeating-linear-gradient(45deg,
    rgba(255,255,255,0.0) 0,
    rgba(255,255,255,0.0) 8px,
    rgba(255,255,255,0.12) 8px,
    rgba(255,255,255,0.12) 16px);
  border-radius: 4px;
  animation: p-stripes 1s linear infinite;
}
@keyframes p-stripes {
  from { background-position: 0 0; }
  to   { background-position: 16px 0; }
}
.p-prog-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: 14px;
  font-size: 11.5px; color: var(--muted);
}
.p-prog-row .l { display: flex; align-items: center; gap: 12px; font-family: var(--mono); }
.p-prog-row .l .pct { color: var(--orange); font-weight: 700; font-size: 14px; font-variant-numeric: tabular-nums; }
.p-prog-row .r { display: flex; align-items: center; gap: 14px; font-family: var(--mono); }
.p-prog-row .r b { color: var(--t); font-weight: 500; font-variant-numeric: tabular-nums; }

/* STEPS */
.p-steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.p-step {
  position: relative;
  padding: 12px 12px 14px;
  background: var(--bg); border: 1px solid var(--bd);
  border-radius: 7px;
  display: flex; flex-direction: column; gap: 7px;
}
.p-step::after {
  content: ''; position: absolute; right: -7px; top: 50%;
  width: 8px; height: 1px; background: var(--bd2);
  transform: translateY(-50%);
}
.p-step:last-child::after { display: none; }
.p-step .top {
  display: flex; align-items: center; justify-content: space-between;
}
.p-step .top .stat {
  width: 18px; height: 18px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 9px; font-weight: 700;
}
.p-step .top .stat.done { background: var(--green); color: #0E1014; }
.p-step .top .stat.active {
  background: var(--orange); color: #0E1014;
  animation: p-pulse 1.5s ease-in-out infinite;
}
.p-step .top .stat.pending { background: transparent; border: 1.5px solid var(--bd2); color: var(--faint); }
@keyframes p-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(245,144,66,0.4); }
  50% { box-shadow: 0 0 0 6px rgba(245,144,66,0); }
}
.p-step .top .num {
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  letter-spacing: 0.06em;
}
.p-step .name {
  font-size: 13px; font-weight: 600; color: var(--t); letter-spacing: -0.005em;
}
.p-step.pending .name { color: var(--muted); }
.p-step.active .name { color: var(--orange); }
.p-step.done .name { color: var(--green); }
.p-step .meta {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  display: flex; align-items: center; gap: 6px;
  font-variant-numeric: tabular-nums;
}
.p-step .meta b { color: var(--t2); font-weight: 500; }
.p-step .ministep {
  font-family: var(--mono); font-size: 10px; color: var(--orange);
  display: flex; align-items: center; gap: 5px;
}
.p-step .ministep .dot { width: 4px; height: 4px; border-radius: 50%; background: var(--orange); }

/* Body: log + sidebar */
.p-body {
  flex: 1; padding: 18px 24px 24px;
  display: grid; grid-template-columns: 1fr 320px;
  gap: 18px; overflow: hidden;
}
.p-log {
  background: #0A0B0E; border: 1px solid var(--bd);
  border-radius: 7px; overflow: hidden;
  display: flex; flex-direction: column;
}
.p-log .head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--bd);
  font-size: 11.5px;
}
.p-log .head .l { display: flex; align-items: center; gap: 8px; color: var(--t2); font-weight: 500; }
.p-log .head .l .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: p-pulse-g 1.5s ease infinite; }
@keyframes p-pulse-g {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.p-log .head .r {
  display: flex; align-items: center; gap: 4px;
  color: var(--muted); font-family: var(--mono); font-size: 10.5px;
}
.p-log .lines {
  flex: 1; overflow-y: auto; padding: 10px 14px;
  font-family: var(--mono); font-size: 11.5px; line-height: 1.5;
  color: var(--t2);
}
.p-log .l-row {
  display: grid; grid-template-columns: 72px 18px 1fr;
  gap: 8px; padding: 2px 0; align-items: baseline;
}
.p-log .l-row .t { color: var(--faint); font-size: 10.5px; }
.p-log .l-row .lv { font-size: 9.5px; padding: 0 4px; border-radius: 3px;
  height: 14px; display: inline-flex; align-items: center; justify-content: center;
  letter-spacing: 0.04em;
}
.p-log .l-row .lv.i { background: var(--bd); color: var(--t2); }
.p-log .l-row .lv.s { background: var(--green-bg); color: var(--green); }
.p-log .l-row .lv.w { background: var(--orange-bg); color: var(--orange); }
.p-log .l-row .lv.d { background: var(--ac-bg); color: var(--ac); }
.p-log .l-row .m { color: var(--t2); }
.p-log .l-row .m .v { color: var(--t); }
.p-log .l-row .m b { color: var(--ac); font-weight: 500; }

/* Right side: stats + queue */
.p-side { display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }
.p-stats {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 7px; padding: 14px;
}
.p-stats h3 {
  margin: 0 0 10px; font-size: 11px; font-weight: 500; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.p-stats .grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.p-stats .stat .v {
  font-family: var(--mono); font-size: 17px; color: var(--t);
  font-variant-numeric: tabular-nums; font-weight: 600;
}
.p-stats .stat .v.warn { color: var(--orange); }
.p-stats .stat .v.ok { color: var(--green); }
.p-stats .stat .k {
  font-size: 10.5px; color: var(--muted);
  margin-top: 2px;
}

.p-queue {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 7px; padding: 14px;
}
.p-queue h3 {
  margin: 0 0 10px; display: flex; align-items: center; justify-content: space-between;
  font-size: 11px; font-weight: 500; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.p-queue h3 a { color: var(--ac); cursor: pointer; font-family: var(--mono); }
.p-queue .item {
  display: grid; grid-template-columns: 22px 1fr auto;
  gap: 8px; padding: 7px 0; align-items: center;
  border-top: 1px solid var(--bd);
  font-size: 12px;
}
.p-queue .item:first-of-type { border-top: none; }
.p-queue .item .dot {
  width: 8px; height: 8px; border-radius: 50%; justify-self: center;
}
.p-queue .item.done .dot { background: var(--green); }
.p-queue .item.queued .dot { background: var(--bd2); border: 1.5px solid var(--bd2); }
.p-queue .item.proc .dot { background: var(--orange); }
.p-queue .item .nm {
  display: flex; align-items: center; gap: 6px;
  color: var(--t2);
}
.p-queue .item .nm .filmdot { width: 6px; height: 6px; border-radius: 50%; }
.p-queue .item.done .nm { color: var(--t); }
.p-queue .item.proc .nm { color: var(--orange); font-weight: 500; }
.p-queue .item .when {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}

/* RIGHT PANE — Active step detail */
.p-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.p-rp .head {
  padding: 13px 16px; border-bottom: 1px solid var(--bd);
}
.p-rp .head h3 {
  margin: 0; display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; color: var(--t);
}
.p-rp .head h3 .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--orange); }
.p-rp .head .sub {
  font-size: 11.5px; color: var(--muted); margin-top: 3px;
}
.p-rp .inner { padding: 14px 16px 16px; overflow-y: auto; flex: 1; }

.p-rp .what { font-size: 12.5px; color: var(--t2); line-height: 1.55; margin-bottom: 14px; }
.p-rp .what b { color: var(--t); }

.p-rp .substeps {
  display: flex; flex-direction: column; gap: 7px;
  margin-bottom: 14px;
}
.p-rp .substep {
  display: grid; grid-template-columns: 16px 1fr auto;
  gap: 8px; align-items: center;
  padding: 7px 9px; background: var(--bg); border: 1px solid var(--bd);
  border-radius: 5px;
  font-size: 12px;
}
.p-rp .substep .icn { display: flex; align-items: center; color: var(--green); }
.p-rp .substep.active .icn { color: var(--orange); }
.p-rp .substep.pending .icn { color: var(--faint); }
.p-rp .substep .v {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.p-rp .sect-head {
  display: flex; align-items: baseline; justify-content: space-between;
  margin: 18px 0 8px;
  font-size: 11px; color: var(--muted); font-weight: 500;
  letter-spacing: 0.04em;
}
.p-rp .gpu-card {
  padding: 12px 14px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
}
.p-rp .gpu-row {
  display: grid; grid-template-columns: 1fr 50px;
  gap: 10px; align-items: center; margin-bottom: 7px;
  font-family: var(--mono); font-size: 11px;
}
.p-rp .gpu-row .lab { color: var(--muted); }
.p-rp .gpu-row .v { text-align: right; color: var(--t); }
.p-rp .gpu-row .track {
  grid-column: 1 / 3; height: 4px; background: var(--bd); border-radius: 2px; position: relative;
  margin-top: -2px; margin-bottom: 6px;
}
.p-rp .gpu-row .track::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac); border-radius: 2px;
}
`;

function ScreenProc() {
  const films = window.FILMS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  // Aruanda is currently processing at ~78% — on step 3 (Visual analysis)
  const activeFilm = byId.aruanda;
  const stepIdx = 2; // 0:Frames 1:Cenas 2:Visual 3:Embeddings 4:Descrições
  const overallPct = 78;
  const stepPct = 64;

  const steps = [
    { name: 'Frames',      sub: 'extração',  v: '14.892',  k: 'quadros',     stat: 'done' },
    { name: 'Cenas',       sub: 'detecção',  v: '94',      k: 'cenas',       stat: 'done' },
    { name: 'Visual',      sub: 'análise',   v: '60 / 94', k: 'cenas',       stat: 'active', mini: 'YOLOv8 + MTCNN · 4.2s/cena' },
    { name: 'Embeddings',  sub: 'CLIP-L/14', v: '0',       k: 'cenas',       stat: 'pending' },
    { name: 'Descrições',  sub: 'moondream', v: '0',       k: 'cenas',       stat: 'pending' },
  ];

  const logLines = [
    {t:'14:32:18', l:'info',  m:<><span>visual.detect</span> · cena <b>60</b>/94 · <span className="v">3 objetos</span> · 4.31s</>},
    {t:'14:32:14', l:'info',  m:<><span>visual.detect</span> · cena <b>59</b>/94 · <span className="v">7 objetos · 1 face</span> · 4.18s</>},
    {t:'14:32:09', l:'debug', m:<><span>mtcnn</span> · cena <b>59</b> · face_count=<span className="v">1</span> · score=<span className="v">0.94</span></>},
    {t:'14:32:09', l:'info',  m:<><span>visual.detect</span> · cena <b>58</b>/94 · <span className="v">5 objetos</span> · 4.42s</>},
    {t:'14:32:05', l:'info',  m:<><span>visual.detect</span> · cena <b>57</b>/94 · <span className="v">2 objetos</span> · 4.07s</>},
    {t:'14:32:01', l:'info',  m:<><span>yolo</span> · cena <b>57</b> · classes=[person, cow, sky]</>},
    {t:'14:31:57', l:'info',  m:<><span>visual.detect</span> · cena <b>56</b>/94 · <span className="v">4 objetos</span> · 4.25s</>},
    {t:'14:31:53', l:'warn',  m:<><span>visual.detect</span> · cena <b>55</b> · keyframe blur=<span className="v">0.42</span> · alta · re-extraindo</>},
    {t:'14:31:49', l:'info',  m:<><span>visual.detect</span> · cena <b>54</b>/94 · <span className="v">8 objetos · 3 faces</span> · 4.11s</>},
    {t:'14:31:45', l:'info',  m:<><span>visual.detect</span> · cena <b>53</b>/94 · <span className="v">2 objetos</span> · 4.04s</>},
    {t:'14:31:41', l:'info',  m:<><span>visual.detect</span> · cena <b>52</b>/94 · <span className="v">6 objetos · 1 face</span> · 4.33s</>},
    {t:'14:31:36', l:'ok',    m:<><span>cena.detect</span> · concluído · 94 cenas (mediana 2.4s)</>},
    {t:'14:30:14', l:'info',  m:<><span>cena.detect</span> · começando detecção em 14892 quadros</>},
    {t:'14:28:09', l:'ok',    m:<><span>frames.extract</span> · concluído · 14892 quadros · 96.4 fps</>},
    {t:'14:26:50', l:'info',  m:<><span>frames.extract</span> · iniciando · 21:00 min · 25fps</>},
    {t:'14:26:48', l:'info',  m:<><span>pipeline.start</span> · aruanda.mp4 · CLIP-L/14 + md2 + YOLOv8 + MTCNN</>},
  ];

  const lvCls = (l) => ({info:'i', debug:'d', warn:'w', ok:'s'}[l] || 'i');

  return (
    <>
      <section className="p-cp">
        <div className="p-top">
          <div className="row1">
            <h1>
              <span className="ic"><I.proc /></span>
              Processamento
              <span className="pip">1 ativo</span>
            </h1>
            <div className="acts">
              <button className="fx-btn secondary"><I.plus /> Novo filme</button>
              <button className="fx-btn secondary"><I.upload /> Importar lote</button>
              <button className="fx-btn primary"><I.play /> Iniciar processamento</button>
            </div>
          </div>

          <div className="p-active">
            <div className="head">
              <div className="l">
                <div className="thumb" style={{backgroundImage:`url(keyframes/kf-17-smoke.jpg)`}}></div>
                <div className="info">
                  <h2>
                    <span className="dot"></span>
                    {activeFilm.title}
                    <span className="stat">processando</span>
                  </h2>
                  <div className="sub">
                    <span><b>{activeFilm.year}</b></span>
                    <span>·</span>
                    <span>dir. {activeFilm.director}</span>
                    <span>·</span>
                    <span>{activeFilm.runtime} min</span>
                    <span>·</span>
                    <span style={{fontFamily:'var(--mono)'}}>iniciado às 14:26:48 · há 5min32s</span>
                  </div>
                </div>
              </div>
              <div className="r">
                <button className="fx-btn ghost"><I.expand /></button>
                <button className="fx-btn secondary">Pausar</button>
                <button className="fx-btn secondary" style={{color: FX.red, borderColor: FX.red}}>Cancelar</button>
              </div>
            </div>

            <div className="p-pbar" style={{'--p': overallPct + '%'}}></div>
            <div className="p-prog-row">
              <div className="l">
                <span className="pct">{overallPct}%</span>
                <span>concluído</span>
              </div>
              <div className="r">
                <span>etapa <b>3 / 5</b></span>
                <span>·</span>
                <span><b>60 / 94</b> cenas (etapa)</span>
                <span>·</span>
                <span>throughput <b>14.3</b> cenas/min</span>
                <span>·</span>
                <span>ETA <b>~2min 24s</b></span>
              </div>
            </div>

            {/* STEPS */}
            <div className="p-steps">
              {steps.map((s, i) => (
                <div key={i} className={'p-step ' + s.stat}>
                  <div className="top">
                    <span className={'stat ' + s.stat}>
                      {s.stat === 'done' && <I.check />}
                      {s.stat === 'active' && i+1}
                      {s.stat === 'pending' && (i+1)}
                    </span>
                    <span className="num">0{i+1}/05</span>
                  </div>
                  <div className="name">{s.name}</div>
                  <div className="meta">
                    <span>{s.sub}</span>
                    <span>·</span>
                    <b>{s.v}</b>
                    <span>{s.k}</span>
                  </div>
                  {s.mini && <div className="ministep"><span className="dot"></span>{s.mini}</div>}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="p-body">
          <div className="p-log">
            <div className="head">
              <div className="l">
                <span className="dot"></span>
                <span>Log · em tempo real</span>
                <span style={{fontFamily:'var(--mono)', fontSize:10.5, color: FX.muted}}>aruanda · SSE</span>
              </div>
              <div className="r">
                <button className="fx-icbtn sm"><I.filter /></button>
                <button className="fx-icbtn sm"><I.download /></button>
                <span>auto-scroll · on</span>
              </div>
            </div>
            <div className="lines">
              {logLines.map((ln, i) => (
                <div key={i} className="l-row">
                  <span className="t">{ln.t}</span>
                  <span className={'lv ' + lvCls(ln.l)}>{ln.l}</span>
                  <span className="m">{ln.m}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-side">
            <div className="p-stats">
              <h3>Estatísticas · processo atual</h3>
              <div className="grid">
                <div className="stat">
                  <div className="v">14.892</div>
                  <div className="k">quadros extraídos</div>
                </div>
                <div className="stat">
                  <div className="v ok">94</div>
                  <div className="k">cenas detectadas</div>
                </div>
                <div className="stat">
                  <div className="v warn">60 / 94</div>
                  <div className="k">cenas analisadas</div>
                </div>
                <div className="stat">
                  <div className="v">14.3</div>
                  <div className="k">cenas / min</div>
                </div>
                <div className="stat">
                  <div className="v">12</div>
                  <div className="k">faces detectadas</div>
                </div>
                <div className="stat">
                  <div className="v">128</div>
                  <div className="k">objetos · YOLOv8</div>
                </div>
              </div>
            </div>

            <div className="p-queue">
              <h3>
                <span>Fila · histórico</span>
                <a>ver tudo</a>
              </h3>
              <div className="item done">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.jeca}}></span>
                  Jeca Tatu
                </span>
                <span className="when">há 4d</span>
              </div>
              <div className="item done">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.limite}}></span>
                  Limite
                </span>
                <span className="when">há 2d</span>
              </div>
              <div className="item done">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.cangaceiro}}></span>
                  O Cangaceiro
                </span>
                <span className="when">há 28h</span>
              </div>
              <div className="item done">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.pagador}}></span>
                  O Pagador de Promessas
                </span>
                <span className="when">há 19h</span>
              </div>
              <div className="item done">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.rio40}}></span>
                  Rio, 40 Graus
                </span>
                <span className="when">há 14h</span>
              </div>
              <div className="item proc">
                <span className="dot"></span>
                <span className="nm">
                  <span className="filmdot" style={{background: FX_FILM.aruanda}}></span>
                  Aruanda · etapa 3/5
                </span>
                <span className="when">agora</span>
              </div>
              <div className="item queued">
                <span className="dot"></span>
                <span className="nm" style={{color: FX.muted}}>
                  <span className="filmdot" style={{background: FX.faint}}></span>
                  Vidas Secas (na fila)
                </span>
                <span className="when">aguardando</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* RIGHT PANE — active step detail */}
      <aside className="p-rp">
        <div className="head">
          <h3>
            <span className="dot"></span>
            Etapa 3 · Análise visual
          </h3>
          <div className="sub">YOLOv8 (objetos) + MTCNN (faces) · cena 60 / 94</div>
        </div>
        <div className="inner">
          <p className="what">
            Roda detecção de objetos e faces em cada keyframe para gerar tags automáticas
            (<b>pessoa</b>, <b>animal</b>, <b>veículo</b>, etc.) e contagem de pessoas por cena.
            Saídas alimentam a etapa de embeddings e a de descrições.
          </p>

          <div className="p-rp .sect-head" style={{margin:'0 0 8px', fontSize:11, color: FX.muted, fontWeight: 500, letterSpacing: '0.04em', textTransform: 'uppercase'}}>Sub-etapas</div>
          <div className="substeps">
            <div className="substep">
              <span className="icn"><I.check /></span>
              <span>Carregar pesos YOLOv8 · ultralytics</span>
              <span className="v">0.4s</span>
            </div>
            <div className="substep">
              <span className="icn"><I.check /></span>
              <span>Carregar MTCNN · facenet-pytorch</span>
              <span className="v">0.6s</span>
            </div>
            <div className="substep active">
              <span className="icn"><I.proc /></span>
              <span>Detectar objetos · 60 / 94 cenas</span>
              <span className="v" style={{color: FX.orange}}>3min</span>
            </div>
            <div className="substep pending">
              <span className="icn"><I.circle /></span>
              <span>Detectar faces · 60 / 94 cenas</span>
              <span className="v">~ 2min</span>
            </div>
            <div className="substep pending">
              <span className="icn"><I.circle /></span>
              <span>Persistir tags · 0 / 94 cenas</span>
              <span className="v">~ 15s</span>
            </div>
          </div>

          <div className="sect-head" style={{margin: '14px 0 8px', display:'flex', alignItems:'baseline', justifyContent:'space-between', fontSize:11, color: FX.muted, fontWeight: 500, letterSpacing: '0.04em', textTransform: 'uppercase'}}>
            <span>Recursos · cuda:0</span>
            <span style={{fontFamily:'var(--mono)', color: FX.t, textTransform:'none'}}>14.8 GB / 24 GB</span>
          </div>
          <div className="gpu-card">
            <div className="gpu-row">
              <span className="lab">GPU</span><span className="v">62%</span>
              <span className="track" style={{'--p': '62%', gridColumn: '1/3'}}></span>
            </div>
            <div className="gpu-row">
              <span className="lab">VRAM</span><span className="v">14.8GB</span>
              <span className="track" style={{'--p': '62%', gridColumn: '1/3'}}></span>
            </div>
            <div className="gpu-row">
              <span className="lab">CPU</span><span className="v">38%</span>
              <span className="track" style={{'--p': '38%', gridColumn: '1/3'}}></span>
            </div>
            <div className="gpu-row">
              <span className="lab">RAM</span><span className="v">7.2GB</span>
              <span className="track" style={{'--p': '24%', gridColumn: '1/3'}}></span>
            </div>
          </div>

          <div style={{display:'flex', flexDirection:'column', gap:6, marginTop:16}}>
            <button className="fx-btn secondary" style={{justifyContent:'center'}}><I.expand /> Ver detalhe do log</button>
            <button className="fx-btn secondary" style={{justifyContent:'center'}}><I.settings /> Configurar pipeline</button>
          </div>
        </div>
      </aside>
    </>
  );
}

window.ScreenProc = ScreenProc;
window.PROC_CSS = PROC_CSS;
