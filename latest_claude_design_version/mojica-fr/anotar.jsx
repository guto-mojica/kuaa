// Mojica · Frame.io branch · Anotar screen (single scene focused review)
// Big keyframe with annotation pin overlay + inline comment popup,
// right-side comment thread, playback controls + timeline below.

const ANOTAR_CSS = `
.a-stage {
  display: flex; flex-direction: column;
  background: #000; overflow: hidden;
  min-width: 0;
}

.a-meta {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 18px; background: var(--bg);
  border-bottom: 1px solid var(--bd);
}
.a-meta .l { display: flex; align-items: center; gap: 12px; }
.a-meta .back {
  width: 28px; height: 28px; border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  background: var(--raised); color: var(--t2); cursor: pointer;
  border: 1px solid var(--bd);
}
.a-meta .back:hover { background: var(--hover); color: var(--t); }
.a-meta .filmpath {
  display: flex; align-items: center; gap: 8px;
  font-size: 13.5px; color: var(--t2);
}
.a-meta .filmpath .dot { width: 8px; height: 8px; border-radius: 50%; }
.a-meta .filmpath .seg { color: var(--t2); }
.a-meta .filmpath .seg.cur { color: var(--t); font-weight: 500; }
.a-meta .filmpath .sep { color: var(--faint); }
.a-meta .filmpath .ver {
  font-family: var(--mono); font-size: 11px; padding: 2px 7px;
  border-radius: 4px; background: var(--raised); color: var(--t);
  border: 1px solid var(--bd); cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px;
}
.a-meta .r { display: flex; align-items: center; gap: 6px; }
.a-meta .stat-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px; border-radius: 14px;
  background: var(--green-bg); color: var(--green);
  font-size: 11.5px; font-weight: 500;
}
.a-meta .stat-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

.a-keyframe-wrap {
  flex: 1; display: flex; align-items: center; justify-content: center;
  background: #000; position: relative; overflow: hidden;
  padding: 18px 28px;
}
.a-keyframe {
  max-width: 100%; max-height: 100%;
  aspect-ratio: 4/3;
  width: 100%; height: 100%;
  background-size: contain; background-repeat: no-repeat;
  background-position: center;
  background-color: #000;
  position: relative;
  filter: contrast(1.04);
}
.a-keyframe .pin {
  position: absolute;
  width: 30px; height: 30px; border-radius: 16px 16px 16px 0;
  background: var(--yellow); color: #0E1014;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 14px; font-weight: 800;
  transform: rotate(-12deg);
  box-shadow: 0 0 0 4px rgba(245,200,66,0.22), 0 4px 12px rgba(0,0,0,0.4);
  cursor: pointer;
  z-index: 3;
}
.a-keyframe .pin.alt {
  background: var(--ac); color: #fff;
  box-shadow: 0 0 0 4px rgba(139,123,216,0.22), 0 4px 12px rgba(0,0,0,0.4);
}

.a-keyframe .commentpop {
  position: absolute;
  width: 320px; padding: 14px;
  background: var(--panel); border: 1px solid var(--bd2);
  border-radius: 7px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,0,0,0.3);
  z-index: 4;
  display: flex; flex-direction: column; gap: 10px;
}
.a-keyframe .commentpop .com {
  display: grid; grid-template-columns: 26px 1fr; gap: 9px;
}
.a-keyframe .commentpop .com .av {
  width: 24px; height: 24px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10px; font-weight: 600; color: #fff;
}
.a-keyframe .commentpop .com.ai .av { background: linear-gradient(135deg, var(--ac), var(--ac-dim)); }
.a-keyframe .commentpop .com.curator .av { background: linear-gradient(135deg, var(--pink), #B0432B); }
.a-keyframe .commentpop .com .who {
  display: flex; align-items: baseline; gap: 6px;
  font-size: 11.5px;
}
.a-keyframe .commentpop .com .who .n { color: var(--t); font-weight: 500; }
.a-keyframe .commentpop .com .who .when {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
}
.a-keyframe .commentpop .com .body {
  font-size: 12.5px; line-height: 1.5; color: var(--t2); margin-top: 3px;
  text-wrap: pretty;
}
.a-keyframe .commentpop .com .body .fx-tc { margin-right: 4px; }
.a-keyframe .commentpop .com .replyrow {
  display: flex; gap: 12px; font-size: 11px; color: var(--muted); margin-top: 5px;
}
.a-keyframe .commentpop .com .replyrow a { cursor: pointer; }
.a-keyframe .commentpop .com .replyrow a:hover { color: var(--ac); }

/* Timeline below keyframe */
.a-tl {
  padding: 12px 18px 10px;
  background: var(--bg); border-top: 1px solid var(--bd);
}
.a-tl .scrubrow {
  display: flex; align-items: center; gap: 10px;
}
.a-tl .scrub {
  flex: 1; position: relative; height: 28px;
  background: var(--raised); border-radius: 4px; cursor: pointer;
}
.a-tl .scrub .progress {
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 38%; background: rgba(139,123,216,0.30);
  border-radius: 4px 0 0 4px;
}
.a-tl .scrub .progress::after {
  content: ''; position: absolute; right: -1px; top: 0; bottom: 0; width: 2px;
  background: var(--ac);
}
.a-tl .scrub .marker {
  position: absolute; top: 4px; bottom: 4px;
  width: 6px; border-radius: 2px;
  background: var(--yellow);
  cursor: pointer;
}
.a-tl .scrub .marker.curator { background: var(--pink); }
.a-tl .scrub .marker .label {
  position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
  margin-bottom: 5px;
  font-family: var(--mono); font-size: 9.5px; color: var(--t);
  white-space: nowrap;
  background: var(--bg); padding: 1px 5px; border-radius: 3px;
  border: 1px solid var(--bd);
  opacity: 0; transition: opacity .15s;
}
.a-tl .scrub .marker:hover .label { opacity: 1; }
.a-tl .scrub .av-pip {
  position: absolute; top: -8px;
  width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--bg);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 8.5px; font-weight: 700; color: #fff;
  cursor: pointer;
  transform: translateX(-50%);
}
.a-tl .scrub .av-pip.y { background: var(--yellow); color: #0E1014; }
.a-tl .scrub .av-pip.p { background: var(--pink); color: #fff; }
.a-tl .scrub .av-pip.g { background: var(--green); color: #0E1014; }
.a-tl .tc {
  font-family: var(--mono); font-size: 11.5px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.a-tl .tc b { color: var(--yellow); font-weight: 600; }

.a-tl .ticks {
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  padding: 4px 0 0;
}

/* Player controls */
.a-pl {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 18px; background: var(--bg);
  border-top: 1px solid var(--bd);
}
.a-pl .l { display: flex; align-items: center; gap: 4px; }
.a-pl .c { flex: 1; display: flex; align-items: center; justify-content: center; gap: 16px;
  font-family: var(--mono); font-size: 12.5px; color: var(--muted);
}
.a-pl .c .tc-now { color: var(--t); font-weight: 600; font-variant-numeric: tabular-nums; }
.a-pl .c .tc-tot { color: var(--muted); font-variant-numeric: tabular-nums; }
.a-pl .r { display: flex; align-items: center; gap: 4px; }
.a-pl .icp {
  width: 32px; height: 30px; border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; color: var(--t2); cursor: pointer;
  font-size: 14px;
}
.a-pl .icp:hover { background: var(--hover); color: var(--t); }
.a-pl .icp.primary {
  background: var(--ac); color: #fff;
  width: 36px; height: 30px;
}
.a-pl .icp.primary:hover { background: var(--ac2); }
.a-pl .label {
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  padding: 5px 9px; border-radius: 5px; cursor: pointer;
  display: flex; align-items: center; gap: 5px;
}
.a-pl .label:hover { background: var(--hover); color: var(--t); }
.a-pl .label.on { background: var(--hover); color: var(--t); }
.a-pl .label.on .v { color: var(--ac); }

/* Right comments pane */
.a-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.a-rp .htabs {
  display: flex; align-items: center; padding: 0 12px;
  border-bottom: 1px solid var(--bd); gap: 2px;
}
.a-rp .htab {
  padding: 13px 12px; font-size: 12.5px; color: var(--muted);
  cursor: pointer; position: relative; font-weight: 500;
  display: flex; align-items: center; gap: 6px;
}
.a-rp .htab:hover { color: var(--t); }
.a-rp .htab.on { color: var(--t); }
.a-rp .htab.on::after {
  content: ''; position: absolute; left: 8px; right: 8px; bottom: -1px;
  height: 2px; background: var(--ac); border-radius: 1px;
}
.a-rp .htab .pip {
  font-family: var(--mono); font-size: 10px; padding: 0 5px;
  background: var(--raised); border-radius: 8px; color: var(--t2);
}
.a-rp .htabs .gap { flex: 1; }
.a-rp .htabs .ic { padding: 6px; border-radius: 4px; color: var(--muted); cursor: pointer; display:flex; align-items:center; }
.a-rp .htabs .ic:hover { background: var(--hover); color: var(--t); }

.a-rp .subhead {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; border-bottom: 1px solid var(--bd);
  font-size: 12px;
}
.a-rp .subhead .l {
  display: flex; align-items: center; gap: 8px;
  color: var(--t2);
}
.a-rp .subhead .l .v { color: var(--t); font-weight: 500; }
.a-rp .subhead .l .pip {
  font-family: var(--mono); font-size: 10.5px;
  background: var(--ac); color: #fff; padding: 1px 6px; border-radius: 8px;
  font-weight: 600;
}
.a-rp .subhead .r { display: flex; align-items: center; gap: 2px; color: var(--muted); }

.a-thread { flex: 1; overflow-y: auto; padding: 14px 16px; display: flex; flex-direction: column; gap: 16px; }

.a-com {
  display: grid; grid-template-columns: 30px 1fr; gap: 10px;
  position: relative;
}
.a-com .av {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10.5px; font-weight: 600; color: #fff;
}
.a-com.ai .av { background: linear-gradient(135deg, var(--ac), var(--ac-dim)); }
.a-com.curator .av { background: linear-gradient(135deg, var(--pink), #B0432B); }
.a-com.viewer1 .av { background: linear-gradient(135deg, #5CCB91, #2E8A5D); }
.a-com .bx { display: flex; flex-direction: column; gap: 4px; }
.a-com .who {
  display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap;
  font-size: 12.5px;
}
.a-com .who .n { color: var(--t); font-weight: 500; }
.a-com .who .when { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
.a-com .who .ix { color: var(--faint); font-family: var(--mono); font-size: 10.5px; margin-left: auto; }
.a-com .badge {
  font-family: var(--mono); font-size: 9.5px; padding: 1px 5px;
  border-radius: 3px; background: var(--raised); color: var(--t2);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.a-com.ai .badge { background: var(--ac-bg); color: var(--ac); }
.a-com.pinned .badge { background: var(--yellow-bg); color: var(--yellow); }
.a-com .body {
  font-size: 13px; line-height: 1.55; color: var(--t2);
  text-wrap: pretty;
}
.a-com .attach {
  display: flex; flex-direction: column; gap: 4px; margin-top: 6px;
}
.a-com .attach .att {
  display: flex; align-items: center; gap: 7px; padding: 6px 10px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 5px;
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  cursor: pointer;
}
.a-com .attach .att:hover { border-color: var(--bd2); color: var(--t); }
.a-com .attach .att .ico { color: var(--ac); }
.a-com .attach .att .dl { margin-left: auto; color: var(--muted); }
.a-com .reactrow {
  display: flex; align-items: center; gap: 8px; margin-top: 6px;
}
.a-com .react {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 1px 7px; border-radius: 12px;
  background: var(--raised); border: 1px solid var(--bd);
  font-size: 11px; color: var(--t2); cursor: pointer;
}
.a-com .react.on { background: var(--ac-bg); border-color: var(--ac-dim); color: var(--ac); }
.a-com .react .emoji { font-size: 12px; }
.a-com .replyrow {
  display: flex; align-items: center; gap: 12px; margin-top: 4px;
  font-size: 11.5px; color: var(--muted);
}
.a-com .replyrow a { color: var(--muted); cursor: pointer; }
.a-com .replyrow a:hover { color: var(--ac); }

/* COMPOSER */
.a-rp .composer {
  border-top: 1px solid var(--bd);
  padding: 12px 16px 14px;
  background: var(--panel);
}
.a-rp .composer .meta {
  display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.a-rp .composer .meta .fx-tc { font-weight: 600; }
.a-rp .composer .meta .toolset {
  display: flex; align-items: center; gap: 0; margin-left: auto; color: var(--muted);
}
.a-rp .composer .meta .toolset .t {
  width: 26px; height: 26px; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
}
.a-rp .composer .meta .toolset .t:hover { background: var(--hover); color: var(--t); }
.a-rp .composer .meta .toolset .t.on { background: var(--yellow-bg); color: var(--yellow); }
.a-rp .composer .input-wrap {
  background: var(--bg); border: 1px solid var(--bd);
  border-radius: 7px; padding: 8px 12px;
  display: flex; align-items: flex-end; gap: 8px;
  transition: border-color .12s, box-shadow .12s;
}
.a-rp .composer .input-wrap:focus-within {
  border-color: var(--ac);
  box-shadow: 0 0 0 3px var(--ac-bg-low);
}
.a-rp .composer textarea {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; font-size: 13px; color: var(--t);
  min-height: 36px; max-height: 100px; resize: none;
}
.a-rp .composer textarea::placeholder { color: var(--muted); }
.a-rp .composer .send {
  width: 30px; height: 30px; border-radius: 50%;
  background: var(--ac); color: #fff; border: none;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; flex-shrink: 0;
}
.a-rp .composer .send:hover { background: var(--ac2); }
`;

function ScreenAnotar({ selected, setSelected }) {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  const r = results[selected];
  const f = byId[r.film];

  return (
    <>
      <section className="a-stage">
        {/* META BAR */}
        <div className="a-meta">
          <div className="l">
            <button className="back" title="Voltar"><I.chevL /></button>
            <div className="filmpath">
              <span className="dot" style={{background: FX_FILM[r.film]}}></span>
              <span className="seg">Acervo</span>
              <span className="sep">/</span>
              <span className="seg">{f.title}</span>
              <span className="sep">/</span>
              <span className="seg cur">cena {String(r.cena).padStart(3,'0')}</span>
              <span className="ver">V3 <I.chevD /></span>
            </div>
          </div>
          <div className="r">
            <span className="stat-pill"><span className="dot"></span>indexado</span>
            <button className="fx-icbtn" title="Download"><I.download /></button>
            <button className="fx-icbtn" title="Share"><I.share /></button>
            <button className="fx-icbtn" title="More"><I.more /></button>
            <button className="fx-icbtn" title="Toggle right pane"><I.panelR /></button>
          </div>
        </div>

        {/* KEYFRAME with annotation pin + comment popup */}
        <div className="a-keyframe-wrap">
          <div className="a-keyframe" style={{backgroundImage:`url(${r.kf})`}}>
            <div className="pin" style={{top: '24%', left: '34%'}}>1</div>
            <div className="commentpop" style={{top: '34%', left: '40%'}}>
              <div className="com curator">
                <div className="av">RG</div>
                <div className="bx" style={{flex: 1}}>
                  <div className="who">
                    <span className="n">Rafael Gonzaga</span>
                    <span className="when">há 2h</span>
                  </div>
                  <div className="body">
                    <span className="fx-tc">{r.tc}</span>
                    O diálogo nesta cena é representativo da vertente <b>"campo aberto"</b>. Anexei notas do diretor e referências visuais.
                  </div>
                  <div className="attach" style={{marginTop: 8}}>
                    <div className="att">
                      <span className="ico"><I.doc /></span>
                      <span>notas-jeca-tatu.docx</span>
                      <span className="dl"><I.download /></span>
                    </div>
                    <div className="att">
                      <span className="ico" style={{color: FX.orange}}><I.image /></span>
                      <span>moodboard-retrospectiva.jpg</span>
                      <span className="dl"><I.download /></span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="com ai" style={{borderTop: `1px solid ${FX.border}`, paddingTop: 10}}>
                <div className="av">md</div>
                <div className="bx">
                  <div className="who">
                    <span className="n">moondream-2</span>
                    <span className="when">agora</span>
                  </div>
                  <div className="body">
                    Entendido. Re-gerei a descrição com ênfase no <span style={{color: FX.ac}}>contraste figura-paisagem</span>. Veja sugestão na aba Propriedades.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* TIMELINE SCRUB */}
        <div className="a-tl">
          <div className="scrubrow">
            <span className="tc">00:00 / <b>00:30</b></span>
            <div className="scrub">
              <div className="progress"></div>
              <div className="marker" style={{left: '34%'}}>
                <span className="label">cena 111 · 00:21:58</span>
              </div>
              <div className="marker curator" style={{left: '52%'}}>
                <span className="label">cena 115 · 00:22:15</span>
              </div>
              <div className="av-pip y" style={{left: '34%'}}>R</div>
              <div className="av-pip p" style={{left: '52%'}}>md</div>
              <div className="av-pip g" style={{left: '78%'}}>J</div>
            </div>
          </div>
        </div>

        {/* PLAYBACK CONTROLS */}
        <div className="a-pl">
          <div className="l">
            <button className="icp" title="Anterior"><I.chevL /></button>
            <span className="label">cena <span className="v" style={{color: FX.t}}>{String(r.cena).padStart(3,'0')}</span></span>
            <button className="icp" title="Próximo" style={{transform:'rotate(180deg)'}}><I.chevL /></button>
          </div>
          <div className="c">
            <button className="icp" title="Loop"><I.loop /></button>
            <button className="icp" title="Voltar 5s">‹‹</button>
            <button className="icp primary" title="Pause"><I.pause /></button>
            <button className="icp" title="Avançar 5s">››</button>
            <span className="tc-now">00:00:10</span>
            <span>/</span>
            <span className="tc-tot">00:00:30</span>
          </div>
          <div className="r">
            <button className="icp" title="Volume"><I.volume /></button>
            <span className="label">1.0×</span>
            <span className="label on">HD <span className="v">·</span></span>
            <button className="icp" title="Settings"><I.settings /></button>
            <button className="icp" title="Fullscreen"><I.expand /></button>
          </div>
        </div>
      </section>

      {/* RIGHT — COMMENTS */}
      <aside className="a-rp">
        <div className="htabs">
          <span className="htab on"><I.comment /> Comentários <span className="pip">3</span></span>
          <span className="htab"><I.pin /> Anotações <span className="pip">2</span></span>
          <span className="htab">Propriedades</span>
          <span className="gap"></span>
          <span className="ic"><I.sort /></span>
          <span className="ic"><I.search /></span>
          <span className="ic"><I.more /></span>
        </div>

        <div className="subhead">
          <div className="l">
            <span>Todos os comentários</span>
            <span className="pip">3</span>
          </div>
          <div className="r">
            <button className="fx-icbtn sm" title="Filtrar"><I.filter /></button>
            <button className="fx-icbtn sm" title="Ordenar"><I.sort /></button>
          </div>
        </div>

        <div className="a-thread">
          {/* AI description as first comment */}
          <div className="a-com ai">
            <div className="av">md</div>
            <div className="bx">
              <div className="who">
                <span className="n">moondream-2</span>
                <span className="badge">AI · descrição</span>
                <span className="when">há 4 dias</span>
                <span className="ix">#0</span>
              </div>
              <div className="body">
                <span className="fx-tc">{r.tc}</span>
                {r.desc}
              </div>
              <div className="reactrow">
                <span className="react on"><span className="emoji">✓</span><span>1</span></span>
                <span className="react"><span className="emoji">⚐</span></span>
              </div>
              <div className="replyrow">
                <a>Responder</a>
                <a>Re-gerar</a>
                <a>Editar</a>
              </div>
            </div>
          </div>

          {/* Curator pinned comment */}
          <div className="a-com curator pinned">
            <div className="av">RG</div>
            <div className="bx">
              <div className="who">
                <span className="n">Rafael Gonzaga</span>
                <span className="badge">📍 fixado · {r.tc}</span>
                <span className="when">há 2h</span>
                <span className="ix">#1</span>
              </div>
              <div className="body">
                O diálogo nesta cena é representativo da vertente <b>"campo aberto"</b> em Mazzaropi. Anexei notas do diretor e referências visuais que apareceram na pré-pesquisa.
              </div>
              <div className="attach">
                <div className="att">
                  <span className="ico"><I.doc /></span>
                  <span>notas-jeca-tatu.docx</span>
                  <span className="dl"><I.download /></span>
                </div>
                <div className="att">
                  <span className="ico" style={{color: FX.orange}}><I.image /></span>
                  <span>moodboard-retrospectiva.jpg</span>
                  <span className="dl"><I.download /></span>
                </div>
              </div>
              <div className="reactrow">
                <span className="react"><span className="emoji">👍</span><span>2</span></span>
                <span className="react on"><span className="emoji">📍</span></span>
              </div>
              <div className="replyrow">
                <a>Responder</a>
                <a>Resolver</a>
                <a>Compartilhar</a>
              </div>
            </div>
          </div>

          {/* Viewer reply */}
          <div className="a-com viewer1">
            <div className="av">JR</div>
            <div className="bx">
              <div className="who">
                <span className="n">Júlia Reis</span>
                <span className="when">agora</span>
                <span className="ix">#2</span>
              </div>
              <div className="body">
                Concordo. Talvez vincular também a <span style={{color: FX.ac, cursor:'pointer'}}>PAGD-003</span> que tem o mesmo enquadramento? Achei interessante para uma <span style={{color: FX.ac}}>rima visual</span>.
              </div>
              <div className="reactrow">
                <span className="react"><span className="emoji">👍</span></span>
                <span className="react"><span className="emoji">☺</span></span>
              </div>
              <div className="replyrow">
                <a>Responder</a>
              </div>
            </div>
          </div>
        </div>

        {/* COMPOSER */}
        <div className="composer">
          <div className="meta">
            <span className="fx-tc">00:00:10</span>
            <span style={{fontSize: 11, color: FX.muted}}>Comentando em <b style={{color: FX.t}}>{f.title} · cena {String(r.cena).padStart(3,'0')}</b></span>
            <div className="toolset">
              <span className="t on" title="Pin"><I.pin /></span>
              <span className="t" title="Tag"><I.tag /></span>
              <span className="t" title="Attach"><I.attach /></span>
              <span className="t" title="Emoji"><I.emoji /></span>
            </div>
          </div>
          <div className="input-wrap">
            <textarea placeholder="Deixe um comentário…"></textarea>
            <button className="send"
                    data-tip="Enviar · ⌘⏎"
                    onClick={() => window.ToastBus && window.ToastBus.push({
                      kind: 'success',
                      title: 'Comentário publicado',
                      sub: <>Fixado em <span style={{fontFamily:'var(--mono)', color:'var(--yellow)'}}>00:00:10</span> · 3 espectadores notificados</>,
                    })}>
              <I.send />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

window.ScreenAnotar = ScreenAnotar;
window.ANOTAR_CSS = ANOTAR_CSS;
