// Mojica · Frame.io branch · shared chrome
// TopBar (logo + breadcrumb + tabs + share + viewers + avatar)
// LeftPane (icon rail + film tree + collections + shares) — has compact mode

const CH_CSS = `
/* ─── TOPBAR ────────────────────────────────────────────────────────── */
.ch-top {
  display: grid; grid-template-columns: auto 1fr auto;
  align-items: center; gap: 18px;
  padding: 0 16px;
  border-bottom: 1px solid var(--bd); background: var(--panel);
}
.ch-top .l { display: flex; align-items: center; gap: 12px; }
.ch-top .brand { display: flex; align-items: center; gap: 9px; padding: 5px 9px; border-radius: 6px; cursor: pointer; }
.ch-top .brand:hover { background: var(--hover); }
.ch-top .brand .n {
  font-weight: 600; font-size: 14.5px; color: var(--t); letter-spacing: -0.01em;
}
.ch-top .div { width: 1px; height: 22px; background: var(--bd); }
.ch-top .crumb {
  display: flex; align-items: center; gap: 8px;
  font-size: 13.5px; color: var(--t2);
}
.ch-top .crumb .seg { padding: 2px 4px; cursor: pointer; border-radius: 3px; }
.ch-top .crumb .seg:hover { background: var(--hover); color: var(--t); }
.ch-top .crumb .seg.cur {
  color: var(--t); font-weight: 500;
}
.ch-top .crumb .sep { color: var(--faint); font-size: 12px; }
.ch-top .crumb .ver {
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  background: var(--raised); padding: 2px 7px 2px 8px; border-radius: 4px;
  display: inline-flex; align-items: center; gap: 5px; cursor: pointer;
  border: 1px solid var(--bd);
}
.ch-top .crumb .ver:hover { border-color: var(--bd2); }

/* TABS — centered chip group, Frame.io style */
.ch-top .tabs {
  display: flex; align-items: center; gap: 2px;
  background: var(--raised); border-radius: 7px; padding: 3px;
  justify-self: center;
}
.ch-top .tab {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 12px; border-radius: 5px;
  font-size: 12.5px; color: var(--t2); cursor: pointer;
  font-weight: 500; letter-spacing: -0.003em;
  border: none; background: transparent;
  white-space: nowrap;
  transition: background .12s, color .12s;
}
.ch-top .tab:hover { color: var(--t); }
.ch-top .tab.on {
  background: var(--hover); color: var(--t);
  box-shadow: 0 1px 0 var(--bd2);
}
.ch-top .tab .ico { display: flex; align-items: center; opacity: 0.8; }
.ch-top .tab.on .ico { opacity: 1; color: var(--ac); }
.ch-top .tab .pip {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 16px; padding: 0 5px; height: 16px; border-radius: 8px;
  background: var(--orange); color: #0E1014; font-size: 10px;
  font-weight: 600; font-family: var(--mono); margin-left: 1px;
}

/* RIGHT cluster */
.ch-top .r { display: flex; align-items: center; gap: 6px; }
.ch-top .viewers {
  display: flex; align-items: center; gap: 0;
  padding: 3px 7px 3px 5px; border-radius: 16px;
  background: var(--pink-bg);
  cursor: pointer;
  margin-right: 4px;
}
.ch-top .viewers .stack { display: flex; align-items: center; }
.ch-top .viewers .av {
  width: 20px; height: 20px; border-radius: 50%;
  border: 1.5px solid var(--panel);
  background: var(--raised); margin-left: -6px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  color: var(--t);
}
.ch-top .viewers .av:first-child { margin-left: 0; }
.ch-top .viewers .av.a1 { background: linear-gradient(135deg, #FF8E72, #C24A2E); color: #fff; }
.ch-top .viewers .av.a2 { background: linear-gradient(135deg, #8B7BD8, #5C4FA8); color: #fff; }
.ch-top .viewers .av.a3 { background: linear-gradient(135deg, #5CCB91, #2E8A5D); color: #fff; }
.ch-top .viewers .c {
  display: flex; align-items: center; gap: 4px;
  font-size: 11.5px; font-weight: 600; color: var(--pink);
  padding-left: 6px; font-family: var(--mono);
}

.ch-top .myav {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, var(--ac), var(--pink));
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  color: #FFFFFF; cursor: pointer; letter-spacing: 0;
}

/* ─── BODY LAYOUT ───────────────────────────────────────────────────── */
.ch-body {
  display: grid; grid-template-columns: 56px 248px 1fr;
  height: 100%; overflow: hidden;
}
.ch-body.compact-lp {
  grid-template-columns: 56px 1fr;
}
.ch-body.with-right {
  grid-template-columns: 56px 248px 1fr 380px;
}
.ch-body.compact-lp.with-right {
  grid-template-columns: 56px 1fr 380px;
}

/* ─── ICON RAIL ─────────────────────────────────────────────────────── */
.ch-rail {
  border-right: 1px solid var(--bd); background: var(--bg);
  display: flex; flex-direction: column; align-items: center;
  padding: 12px 0;
}
.ch-rail .stack { display: flex; flex-direction: column; gap: 4px; }
.ch-rail .spacer { flex: 1; }
.ch-rail .ic {
  width: 36px; height: 36px; border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); cursor: pointer; position: relative;
  background: transparent; border: none;
  transition: background .12s, color .12s;
}
.ch-rail .ic:hover { background: var(--hover); color: var(--t); }
.ch-rail .ic.on { background: var(--ac-bg); color: var(--ac); }
.ch-rail .ic .nb {
  position: absolute; top: 4px; right: 4px;
  min-width: 14px; height: 14px; padding: 0 3px;
  border-radius: 8px; background: var(--pink); color: #0E1014;
  font-size: 9px; font-weight: 700; font-family: var(--mono);
  display: flex; align-items: center; justify-content: center;
  border: 2px solid var(--bg);
  line-height: 1;
}

/* ─── FILM/COLLECTION TREE PANE ─────────────────────────────────────── */
.ch-lp {
  border-right: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.ch-lp .scroll { flex: 1; overflow-y: auto; padding-bottom: 8px; }
.ch-lp .filter {
  margin: 12px 12px 6px; padding: 6px 10px;
  background: var(--bg); border: 1px solid var(--bd);
  border-radius: 6px;
  display: flex; align-items: center; gap: 8px;
  font-size: 12px;
}
.ch-lp .filter input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--t);
}
.ch-lp .filter input::placeholder { color: var(--muted); }
.ch-lp .filter .ico { color: var(--muted); }
.ch-lp .filter .kbd {
  font-family: var(--mono); font-size: 10px; padding: 0 4px;
  border: 1px solid var(--bd2); border-radius: 3px; color: var(--muted);
}

.ch-film {
  display: grid; grid-template-columns: 14px 14px 1fr auto;
  align-items: center; gap: 7px;
  padding: 5px 12px; cursor: pointer; position: relative;
  font-size: 13px; color: var(--t2);
}
.ch-film:hover { background: var(--hover); }
.ch-film.active { background: var(--ac-bg); }
.ch-film.active .name { color: var(--t); font-weight: 500; }
.ch-film.has-sel { background: var(--selected); }
.ch-film .arr { color: var(--muted); display: flex; align-items: center; }
.ch-film.active .arr { color: var(--ac); }
.ch-film .ico { color: var(--muted); display: flex; align-items: center; }
.ch-film.proc .ico { color: var(--orange); }
.ch-film .name { font-size: 13px; }
.ch-film.has-sel .name { color: var(--t); }
.ch-film .ct {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.ch-film .meta {
  grid-column: 3 / 5;
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; color: var(--muted); padding: 4px 0 2px;
}
.ch-film .meta .yr { font-family: var(--mono); min-width: 32px; }
.ch-film .meta .bar {
  flex: 1; height: 3px; background: var(--bd); border-radius: 2px; position: relative;
}
.ch-film .meta .bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p, 0%); background: var(--ac-dim); border-radius: 2px;
}
.ch-film.has-sel .meta .bar::before { background: var(--ac); }
.ch-film.proc .meta .bar::before { background: var(--orange); }
.ch-film .meta .m { font-family: var(--mono); color: var(--t2); }
.ch-film .selptr {
  grid-column: 1 / 5; margin-top: 4px;
  display: flex; align-items: center; gap: 7px;
  padding: 6px 8px; background: var(--bg); border-radius: 5px;
  font-family: var(--mono); font-size: 10.5px; color: var(--ac);
  border: 1px solid var(--ac-dim);
}
.ch-film .selptr .thumb {
  width: 28px; height: 20px; border-radius: 3px;
  background-size: cover; background-position: center;
  flex-shrink: 0;
}
.ch-film .selptr .info {
  flex: 1; display: flex; align-items: center; gap: 7px;
}
.ch-film .selptr .info .arr { color: var(--ac); }

/* Collection row */
.ch-coll {
  display: grid; grid-template-columns: 14px 14px 1fr auto;
  align-items: center; gap: 7px;
  padding: 5px 12px; cursor: pointer; position: relative;
  font-size: 13px; color: var(--t2);
}
.ch-coll:hover { background: var(--hover); }
.ch-coll.active { background: var(--ac-bg); color: var(--ac); }
.ch-coll .ico { display: flex; align-items: center; color: var(--muted); }
.ch-coll.active .ico { color: var(--ac); }
.ch-coll .ct { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }

/* foot */
.ch-lp .foot {
  border-top: 1px solid var(--bd);
  padding: 10px 14px;
  display: flex; align-items: center; justify-content: space-between;
}
.ch-lp .foot .stat {
  display: flex; align-items: center; gap: 7px;
  font-size: 11.5px; color: var(--t2);
}
.ch-lp .foot .stat .dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--green);
}
.ch-lp .foot .info {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
`;

// ──────────────────────────────────────────────────────────────────────
function TopBar({ tab, setTab, breadcrumb }) {
  // breadcrumb: array of { label, cur? } segments
  return (
    <div className="ch-top">
      <div className="l">
        <div className="brand">
          <FXMark size={22} />
          <span className="n">Mojica</span>
        </div>
        <span className="div"></span>
        <div className="crumb">
          {breadcrumb.map((seg, i) => (
            <React.Fragment key={i}>
              {i > 0 && <span className="sep">/</span>}
              {seg.ver ? (
                <span className="ver">{seg.label} <I.chevD /></span>
              ) : (
                <span className={'seg' + (seg.cur ? ' cur' : '')}>{seg.label}</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="tabs">
        {[
          {k:'buscar', l:'Buscar',  ico:<I.search />},
          {k:'cenas',  l:'Cenas',   ico:<I.grid />},
          {k:'anotar', l:'Anotar',  ico:<I.tag />},
          {k:'rimas',  l:'Rimas',   ico:<I.rhymes />},
          {k:'proc',   l:'Processamento', ico:<I.proc />, pip:'1'},
        ].map(t => (
          <button key={t.k}
                  className={'tab' + (tab === t.k ? ' on' : '')}
                  onClick={() => setTab(t.k)}>
            <span className="ico">{t.ico}</span>
            <span>{t.l}</span>
            {t.pip && <span className="pip">{t.pip}</span>}
          </button>
        ))}
      </div>

      <div className="r">
        <button className="fx-icbtn" data-tip="Filtros · ⇧⌘F"><I.filter /></button>
        <button className="fx-icbtn" data-tip="Notificações">
          <I.bell /><span className="nb"></span>
        </button>
        <button className="fx-icbtn" data-tip="Importar mídia · ⌘U"><I.upload /></button>
        <span className="div" style={{margin:'0 4px'}}></span>
        <div className="viewers" data-tip="3 espectadores online" data-tip-pos="bottom">
          <div className="stack">
            <div className="av a1">EA</div>
            <div className="av a2">JR</div>
            <div className="av a3">SP</div>
          </div>
          <span className="c">3</span>
        </div>
        <button className="fx-btn primary"
                style={{padding:'6px 13px'}}
                onClick={() => window.ToastBus && window.ToastBus.push({
                  kind: 'success',
                  title: 'Link de compartilhamento copiado',
                  sub: <><span style={{fontFamily:'var(--mono)',color:'var(--ac)'}}>mojica.local/s/r7g3k9</span> · expira em 7 dias</>,
                })}>
          <I.share /> Compartilhar
        </button>
        <span className="div" style={{margin:'0 4px'}}></span>
        <div className="myav" data-tip="Rafael · Curador" data-tip-pos="bottom">RG</div>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
function IconRail({ tab, setTab }) {
  return (
    <div className="ch-rail">
      <div className="stack">
        <button className="ic" title="Home"><I.home /></button>
        <button className={'ic' + (tab === 'buscar' ? ' on' : '')}
                onClick={() => setTab && setTab('buscar')} title="Buscar"><I.search /></button>
        <button className={'ic' + (tab === 'cenas' ? ' on' : '')}
                onClick={() => setTab && setTab('cenas')} title="Cenas"><I.grid /></button>
        <button className={'ic' + (tab === 'anotar' ? ' on' : '')}
                onClick={() => setTab && setTab('anotar')} title="Anotar"><I.tag /></button>
        <button className={'ic' + (tab === 'rimas' ? ' on' : '')}
                onClick={() => setTab && setTab('rimas')} title="Rimas"><I.rhymes /></button>
      </div>
      <div className="spacer"></div>
      <div className="stack">
        <button className={'ic' + (tab === 'proc' ? ' on' : '')}
                onClick={() => setTab && setTab('proc')} title="Processamento">
          <I.proc /><span className="nb">1</span>
        </button>
        <button className="ic" title="Settings"><I.settings /></button>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
function LeftPane({ selectedFilm, selectedScene }) {
  const films = window.FILMS;
  return (
    <aside className="ch-lp">
      <div className="filter">
        <span className="ico"><I.search /></span>
        <input placeholder="Filtrar acervo…" />
        <span className="kbd">⌘/</span>
      </div>

      <div className="scroll">
        <div className="fx-ph">
          <span>Acervo · Filmes</span>
          <button className="add"><I.plus /></button>
        </div>
        {films.map(f => {
          const isProc = f.id === 'aruanda';
          const isActive = selectedFilm && f.id === selectedFilm.id;
          const hasSel = isActive && selectedScene;
          const cnt = selectedScene && selectedScene.film === f.id ? 1 : 0;
          // mock per-film match progress against current query
          const matchPct = ({jeca:80, limite:25, rio40:35, cangaceiro:55, aruanda:40, pagador:65})[f.id] || 30;
          return (
            <div key={f.id}
                 className={'ch-film' + (isActive ? ' active' : '') + (hasSel ? ' has-sel' : '') + (isProc ? ' proc' : '')}>
              <span className="arr">{isActive ? <I.chevD /> : <I.chevR />}</span>
              <span className="ico">{isProc ? <I.proc /> : <I.film />}</span>
              <span className="name">{f.title}</span>
              <span className="ct">{f.scenes}</span>
              <div className="meta">
                <span className="yr">{f.year}</span>
                <span className="bar" style={{'--p': matchPct + '%'}}></span>
                <span className="m">{Math.round(matchPct/100 * f.scenes)}</span>
              </div>
              {hasSel && (
                <div className="selptr">
                  <span className="thumb" style={{backgroundImage:`url(${selectedScene.kf})`}}></span>
                  <div className="info">
                    <span className="arr">↳</span>
                    <span>cena {String(selectedScene.cena).padStart(3,'0')} · {selectedScene.tc}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        <div className="fx-hr" style={{margin:'10px 12px'}}></div>

        <div className="fx-ph">
          <span>Coleções</span>
          <button className="add"><I.plus /></button>
        </div>
        <div className="ch-coll active">
          <span style={{width:14}}></span>
          <span className="ico"><I.grid /></span>
          <span>Acervo inteiro</span>
          <span className="ct">1.588</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico" style={{color: FX.catExterior}}><I.folder /></span>
          <span>Exteriores rurais</span>
          <span className="ct">142</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico" style={{color: FX.catCartela}}><I.folder /></span>
          <span>Cartelas de título</span>
          <span className="ct">28</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico" style={{color: FX.catDialogo}}><I.folder /></span>
          <span>Diálogos no campo</span>
          <span className="ct">96</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico" style={{color: FX.catInterior}}><I.folder /></span>
          <span>Cenas noturnas</span>
          <span className="ct">73</span>
        </div>

        <div className="fx-hr" style={{margin:'10px 12px'}}></div>

        <div className="fx-ph">
          <span>Compartilhados</span>
          <button className="add"><I.plus /></button>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico"><I.link /></span>
          <span>Curadoria 2026</span>
          <span className="ct">04</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico"><I.link /></span>
          <span>HF Spaces demo</span>
          <span className="ct">live</span>
        </div>
        <div className="ch-coll">
          <span style={{width:14}}></span>
          <span className="ico"><I.globe /></span>
          <span>Avaliação · IRT</span>
          <span className="ct">132</span>
        </div>
      </div>

      <div className="foot">
        <div className="stat">
          <span className="dot"></span>
          <span>Índice ok</span>
        </div>
        <div className="info">1.588 · 8h54m</div>
      </div>
    </aside>
  );
}

window.TopBar = TopBar;
window.IconRail = IconRail;
window.LeftPane = LeftPane;
window.CH_CSS = CH_CSS;
