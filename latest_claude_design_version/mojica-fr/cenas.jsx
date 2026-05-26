// Mojica · Frame.io branch · Cenas screen (full library browse)

const CENAS_CSS = `
.c-cp { display: flex; flex-direction: column; min-width: 0; overflow: hidden; background: var(--bg); }

.c-cp .toolrow {
  display: flex; align-items: center; gap: 6px;
  padding: 12px 24px 12px; border-bottom: 1px solid var(--bd);
}
.c-cp .tool {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 11px; border-radius: 6px;
  font-size: 12.5px; color: var(--t2); cursor: pointer; font-weight: 500;
  background: transparent; border: 1px solid transparent;
}
.c-cp .tool:hover { background: var(--hover); color: var(--t); }
.c-cp .tool .ico { color: var(--muted); font-size: 13px; display: flex; align-items: center; }
.c-cp .tool .v { color: var(--ac); font-weight: 500; }
.c-cp .tool .pip {
  font-family: var(--mono); font-size: 9.5px; padding: 0 5px;
  background: var(--ac-bg); color: var(--ac); border-radius: 8px;
}
.c-cp .toolrow .div { width: 1px; height: 18px; background: var(--bd); margin: 0 2px; }
.c-cp .toolrow .grow { flex: 1; }
.c-cp .toolrow .find {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 6px; padding: 5px 10px;
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: var(--muted); min-width: 240px;
  transition: border-color .12s, background .12s, box-shadow .12s;
}
.c-cp .toolrow .find:focus-within {
  border-color: var(--ac); background: var(--raised);
  box-shadow: 0 0 0 2px var(--ac-bg-low);
}
.c-cp .toolrow .find input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--t); font-size: 12.5px;
}
.c-cp .toolrow .find input::placeholder { color: var(--muted); }

.c-cp .countrow {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 24px 6px;
  font-size: 12.5px;
}
.c-cp .countrow .chev {
  display: flex; align-items: center; color: var(--muted); cursor: pointer;
}
.c-cp .countrow .v {
  color: var(--t); font-family: var(--mono); font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.c-cp .countrow .lab { color: var(--muted); }
.c-cp .countrow .div { width: 1px; height: 14px; background: var(--bd); }

/* GRID */
.c-cp .grid {
  flex: 1; overflow-y: auto;
  padding: 12px 24px 24px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(248px, 1fr));
  gap: 16px 14px;
  align-content: start;
}
.c-cp .group {
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 10px;
  padding: 16px 4px 6px;
  font-size: 13px; color: var(--t2);
  border-bottom: 1px solid var(--bd);
  margin-bottom: 2px;
}
.c-cp .group .chev { color: var(--muted); cursor: pointer; }
.c-cp .group .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.c-cp .group .name { font-weight: 600; color: var(--t); }
.c-cp .group .ct {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  background: var(--raised); padding: 1px 6px; border-radius: 9px;
}
.c-cp .group .meta {
  font-size: 11px; color: var(--muted); margin-left: auto;
  display: flex; align-items: center; gap: 12px;
  font-family: var(--mono);
}

.c-cp .scenecard {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 7px; overflow: hidden;
  cursor: pointer; display: flex; flex-direction: column;
  transition: border-color .12s, transform .12s;
  position: relative;
}
.c-cp .scenecard:hover { border-color: var(--bd2); transform: translateY(-1px); }
.c-cp .scenecard.sel { border-color: var(--ac); box-shadow: 0 0 0 2px var(--ac-bg); }
.c-cp .scenecard .check {
  position: absolute; top: 7px; left: 7px;
  width: 18px; height: 18px; border-radius: 4px;
  background: rgba(14,16,20,0.75); border: 1px solid var(--bd2);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; opacity: 0; transition: opacity .12s;
  color: var(--t); z-index: 2;
}
.c-cp .scenecard:hover .check { opacity: 1; }
.c-cp .scenecard.sel .check {
  opacity: 1; background: var(--ac); border-color: var(--ac); color: #fff;
}
.c-cp .scenecard .kf {
  width: 100%; aspect-ratio: 16/10;
  background: var(--bg) center/cover no-repeat;
  position: relative; filter: contrast(1.04) brightness(0.95);
}
.c-cp .scenecard .kf .bl {
  position: absolute; bottom: 7px; left: 7px;
  display: flex; align-items: center; gap: 4px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 2px 6px; background: rgba(14,16,20,0.78); border-radius: 4px;
}
.c-cp .scenecard .kf .bl .pin { width: 6px; height: 6px; border-radius: 50%; background: var(--yellow); }
.c-cp .scenecard .kf .br {
  position: absolute; bottom: 7px; right: 7px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 2px 6px; background: rgba(14,16,20,0.78); border-radius: 4px;
  font-variant-numeric: tabular-nums;
}
.c-cp .scenecard .kf .ver {
  position: absolute; top: 7px; right: 7px;
  font-family: var(--mono); font-size: 9.5px; color: var(--t);
  padding: 1px 5px; background: rgba(14,16,20,0.78); border-radius: 3px;
  letter-spacing: 0.04em;
}
.c-cp .scenecard .body { padding: 10px 11px 12px; display: flex; flex-direction: column; gap: 6px; }
.c-cp .scenecard .name {
  font-size: 12.5px; font-weight: 600; color: var(--t);
  font-family: var(--mono); letter-spacing: -0.003em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.c-cp .scenecard .sub {
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; color: var(--muted); flex-wrap: wrap;
}
.c-cp .scenecard .sub .dot-f { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.c-cp .scenecard .sub .acts { margin-left: auto; }
.c-cp .scenecard .sub .acts .fx-icbtn { margin: -4px 0; }
.c-cp .scenecard .tipo {
  margin-top: 4px; display: flex; align-items: center; gap: 6px;
}
.c-cp .scenecard .tipo .lab {
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  letter-spacing: 0.06em; text-transform: uppercase; width: 32px;
}
.c-cp .scenecard .tipo .pill {
  font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 500;
}

/* Right pane — selection summary */
.c-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.c-rp .head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-bottom: 1px solid var(--bd);
}
.c-rp .head .l {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; color: var(--t);
}
.c-rp .head .l .badge {
  font-family: var(--mono); font-size: 10.5px;
  background: var(--ac); color: #fff; padding: 1px 7px; border-radius: 9px;
  font-weight: 600;
}
.c-rp .head .r { display: flex; align-items: center; gap: 2px; }
.c-rp .inner { padding: 16px 18px 18px; overflow-y: auto; flex: 1; }
.c-rp .preview {
  width: 100%; aspect-ratio: 16/10; border-radius: 6px;
  background: var(--bg) center/cover no-repeat;
  border: 1px solid var(--bd); filter: contrast(1.05) brightness(0.96);
  position: relative;
}
.c-rp .preview .tipo-pill {
  position: absolute; top: 8px; left: 8px;
}
.c-rp h2 {
  margin: 14px 0 0; font-size: 16px; font-weight: 600; color: var(--t);
  letter-spacing: -0.01em;
}
.c-rp .at {
  display: flex; align-items: center; gap: 7px;
  font-size: 11.5px; color: var(--muted); margin-top: 6px;
}
.c-rp .at .dot { width: 7px; height: 7px; border-radius: 50%; }
.c-rp .at b { color: var(--t2); font-weight: 500; }

.c-rp .props {
  display: grid; grid-template-columns: 90px 1fr;
  gap: 7px 12px; margin-top: 18px;
  padding-top: 14px; border-top: 1px solid var(--bd);
  font-size: 12px;
}
.c-rp .props .k { color: var(--muted); }
.c-rp .props .v { color: var(--t); display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.c-rp .props .v.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

.c-rp .desc-sect {
  margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--bd);
}
.c-rp .desc-sect .lab {
  display: flex; align-items: baseline; justify-content: space-between;
  font-size: 11px; color: var(--muted); font-weight: 500;
  margin-bottom: 7px;
}
.c-rp .desc-sect .lab a { color: var(--ac); cursor: pointer; font-size: 11px; font-family: var(--mono); }
.c-rp .desc-sect p {
  margin: 0; font-size: 12.5px; color: var(--t2); line-height: 1.55;
  text-wrap: pretty;
}

.c-rp .actions {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 16px;
}
.c-rp .actions .fx-btn { justify-content: center; }
`;

// Map a scene to a tipo category based on its tags
function tipoOf(tags, desc) {
  if (desc && desc.toLowerCase().includes('title') || tags.some(t => t.includes('white-writing'))) return 'cartela';
  if (tags.some(t => t.includes('interior')) || tags.some(t => t.includes('baixa-luz'))) return 'interior';
  if (tags.includes('exterior') || tags.some(t => t.includes('rural'))) return 'exterior';
  if (tags.some(t => t.includes('duas-pessoas'))) return 'dialogo';
  return 'transicao';
}

const TIPO_INFO = {
  cartela:    { label: 'Cartela',    color: FX.catCartela,    bg: FX.yellowBg },
  dialogo:    { label: 'Diálogo',    color: FX.catDialogo,    bg: FX.acBg },
  exterior:   { label: 'Exterior',   color: FX.catExterior,   bg: FX.greenBg },
  interior:   { label: 'Interior',   color: FX.catInterior,   bg: FX.orangeBg },
  transicao:  { label: 'Transição',  color: FX.catTransicao,  bg: 'rgba(120,126,138,0.14)' },
};

function ScreenCenas({ selected, setSelected }) {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  // Mock a fuller catalog: 16 scenes spread across films
  // (reuses results array + a few extra mocked entries to feel like a real library)
  const scenes = React.useMemo(() => {
    const ext = [
      { id:101, film:'jeca',       kf:'keyframes/kf-01-title.jpg',     tc:'00:00:00:00', cena:1,   desc:'Title card displayed on a blackboard, names of the cast and crew.', tags:['cartela','title-card','white-writing','interior'] },
      { id:102, film:'jeca',       kf:'keyframes/kf-09-bed.jpg',       tc:'01:23:07:23', cena:374, desc:'A man on a bed inside a dim room, light filtering through a doorway.', tags:['interior','baixa-luz','close-up','noite'] },
      { id:103, film:'limite',     kf:'keyframes/kf-15-night-fence.jpg', tc:'00:35:59:18', cena:172, desc:'Night fence in front of a wooden house, faint moonlight on the planks.', tags:['exterior','noite','fence'] },
      { id:104, film:'rio40',      kf:'keyframes/kf-13-conversation.jpg', tc:'00:23:52:20', cena:135, desc:'Three figures in mid conversation inside a sunlit interior, plants visible.', tags:['interior','dia','duas-pessoas','tres-pessoas'] },
      { id:105, film:'cangaceiro', kf:'keyframes/kf-14-brinquinho.jpg', tc:'01:28:21:03', cena:400, desc:'An ornate small structure labelled "Brinquinho\u2019s Home" in dust and weeds.', tags:['exterior','transicao','dia'] },
      { id:106, film:'aruanda',    kf:'keyframes/kf-17-smoke.jpg',     tc:'01:14:22:17', cena:326, desc:'Smoke billowing from a smouldering wooden frame in midday haze.', tags:['exterior','transicao','smoke','dia'] },
      { id:107, film:'pagador',    kf:'keyframes/kf-18-night-fire.jpg', tc:'00:50:26:21', cena:222, desc:'A small thatched hut burns under a black night sky, embers spitting.', tags:['exterior','noite','fire'] },
    ];
    return [...results, ...ext];
  }, [results]);

  // group by film
  const groups = React.useMemo(() => {
    const m = {};
    scenes.forEach(s => { (m[s.film] = m[s.film] || []).push(s); });
    return m;
  }, [scenes]);

  const selectedScene = scenes[selected] || scenes[0];
  const selFilm = byId[selectedScene.film];
  const selTipo = TIPO_INFO[tipoOf(selectedScene.tags, selectedScene.desc)];

  return (
    <>
      <section className="c-cp">
        <div className="toolrow">
          <span className="tool"><span className="ico"><I.appearance /></span>Aparência</span>
          <span className="tool"><span className="ico"><I.fields /></span>Campos <span className="pip">2</span></span>
          <span className="tool"><span className="ico"><I.filter /></span>Filtros <span className="pip">2</span></span>
          <span className="tool"><span className="ico"><I.group /></span>Agrupado por <span className="v">Filme</span></span>
          <span className="tool"><span className="ico"><I.sort /></span>Ordenado por <span className="v">Timecode</span></span>
          <span className="div"></span>
          <span className="tool"><span className="ico"><I.plus /></span>Adicionar filtro</span>
          <span className="grow"></span>
          <div className="find">
            <span style={{display:'flex', color: FX.muted}}><I.search /></span>
            <input placeholder="Buscar em Acervo inteiro…" />
          </div>
        </div>

        <div className="countrow">
          <span className="chev"><I.chevD /></span>
          <span className="v">1.588</span><span className="lab">cenas</span>
          <span className="div"></span>
          <span className="v">6</span><span className="lab">filmes</span>
          <span className="div"></span>
          <span className="v">8h 54m</span><span className="lab">runtime</span>
          <span className="div"></span>
          <span className="v">3.1 GB</span><span className="lab">keyframes</span>
        </div>

        <div className="grid">
          {films.map(f => {
            const fScenes = groups[f.id] || [];
            if (fScenes.length === 0) return null;
            return (
              <React.Fragment key={f.id}>
                <div className="group">
                  <span className="chev"><I.chevD /></span>
                  <span className="dot" style={{background: FX_FILM[f.id]}}></span>
                  <span className="name">{f.title}</span>
                  <span style={{fontSize:11, color: FX.muted}}>{f.year} · {f.director}</span>
                  <span className="ct">{fScenes.length} / {f.scenes}</span>
                  <span className="meta">
                    <span>runtime {f.runtime}m</span>
                  </span>
                </div>
                {fScenes.map((s) => {
                  const idx = scenes.indexOf(s);
                  const tipo = TIPO_INFO[tipoOf(s.tags, s.desc)];
                  const cmt = (idx % 4 === 0) ? 3 : (idx % 3 === 0) ? 1 : 0;
                  return (
                    <article key={s.id}
                             className={'scenecard' + (idx === selected ? ' sel' : '')}
                             onClick={() => setSelected(idx)}>
                      <span className="check">{idx === selected && <I.check />}</span>
                      <div className="kf" style={{backgroundImage:`url(${s.kf})`}}>
                        {cmt > 0 && <span className="bl"><span className="pin"></span>{cmt}</span>}
                        <span className="br">{s.tc.slice(0,8)}</span>
                        {idx % 5 === 0 && <span className="ver">V{(idx % 3) + 1}</span>}
                      </div>
                      <div className="body">
                        <div className="name">{f.id}_cena_{String(s.cena).padStart(3,'0')}.mp4</div>
                        <div className="sub">
                          <span className="dot-f" style={{background: FX_FILM[f.id]}}></span>
                          <span>{f.director.split(' ').slice(-1)[0]}</span>
                          <span>·</span>
                          <span style={{fontFamily:'var(--mono)'}}>{f.year}</span>
                          <span className="acts">
                            <button className="fx-icbtn sm" onClick={(e)=>e.stopPropagation()}><I.more /></button>
                          </span>
                        </div>
                        <div className="tipo">
                          <span className="lab">Tipo</span>
                          <span className="pill" style={{background: tipo.bg, color: tipo.color}}>
                            {tipo.label}
                          </span>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
      </section>

      <aside className="c-rp">
        <div className="head">
          <div className="l">
            <span className="badge">1</span>
            <span>Item selecionado</span>
          </div>
          <div className="r">
            <button className="fx-icbtn sm" title="Anotar"><I.tag /></button>
            <button className="fx-icbtn sm" title="Compartilhar"><I.share /></button>
            <button className="fx-icbtn sm" title="Baixar"><I.download /></button>
            <button className="fx-icbtn sm" title="Mais"><I.more /></button>
          </div>
        </div>

        <div className="inner">
          <div className="preview" style={{backgroundImage:`url(${selectedScene.kf})`}}>
            <span className="tipo-pill fx-pill" style={{background: selTipo.bg, color: selTipo.color}}>
              {selTipo.label}
            </span>
          </div>

          <h2>cena {String(selectedScene.cena).padStart(3,'0')} · {selFilm.title}</h2>
          <div className="at">
            <span className="dot" style={{background: FX_FILM[selFilm.id]}}></span>
            <b>{selFilm.title}</b>
            <span>·</span>
            <span>{selFilm.year}</span>
            <span>·</span>
            <span>{selFilm.director}</span>
          </div>

          <div className="props">
            <span className="k">Timecode</span>
            <span className="v mono">{selectedScene.tc}</span>

            <span className="k">Duração</span>
            <span className="v mono">~ 4.2 s</span>

            <span className="k">Tipo</span>
            <span className="v">
              <span className="fx-pill" style={{background: selTipo.bg, color: selTipo.color}}>
                {selTipo.label}
              </span>
            </span>

            <span className="k">Tags</span>
            <span className="v">
              {selectedScene.tags.slice(0, 4).map((t, i) => (
                <span key={i} className="fx-pill">{t}</span>
              ))}
              {selectedScene.tags.length > 4 && (
                <span style={{fontFamily:'var(--mono)', fontSize:10.5, color: FX.muted}}>
                  +{selectedScene.tags.length - 4}
                </span>
              )}
            </span>

            <span className="k">Status</span>
            <span className="v">
              <span className="fx-pill green"><span className="dot"></span>indexado</span>
            </span>

            <span className="k">Anotações</span>
            <span className="v mono">2</span>

            <span className="k">Última edição</span>
            <span className="v">há 2h · RG</span>

            <span className="k">Arquivo</span>
            <span className="v mono" style={{color: FX.muted, fontSize: 11}}>
              {selFilm.id}_cena_{String(selectedScene.cena).padStart(3,'0')}.mp4
            </span>
          </div>

          <div className="desc-sect">
            <div className="lab">
              <span>Descrição · moondream-2</span>
              <a>editar</a>
            </div>
            <p>{selectedScene.desc}</p>
          </div>

          <div className="actions">
            <button className="fx-btn primary"><I.play /> Anotar cena</button>
            <button className="fx-btn secondary"><I.rhymes /> Rimas visuais</button>
            <button className="fx-btn secondary" style={{gridColumn:'1 / 3'}}><I.share /> Compartilhar</button>
          </div>
        </div>
      </aside>
    </>
  );
}

window.ScreenCenas = ScreenCenas;
window.CENAS_CSS = CENAS_CSS;
window.tipoOf = tipoOf;
window.TIPO_INFO = TIPO_INFO;
