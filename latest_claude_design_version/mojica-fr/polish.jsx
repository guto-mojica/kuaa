// Mojica · Polish layer
// Command palette, keyboard help overlay, toasts, tooltip CSS, and
// shared transition / focus / press polish. Mounted at the app root.

const POLISH_CSS = `
/* ─── Smooth tab cross-fade ────────────────────────────────────────── */
@keyframes p-fade-in {
  from { opacity: 0; transform: translateY(2px); }
  to   { opacity: 1; transform: translateY(0); }
}
.ch-body > *:not(.ch-rail):not(.ch-lp) {
  animation: p-fade-in 220ms cubic-bezier(.2,.7,.3,1);
  animation-fill-mode: both;
}

/* ─── Focus rings (keyboard nav only) ──────────────────────────────── */
.fx-app :focus { outline: none; }
.fx-app :focus-visible {
  outline: 2px solid var(--ac);
  outline-offset: 2px;
  border-radius: 4px;
}

/* ─── Press states on buttons (subtle compress) ────────────────────── */
.fx-btn:active { transform: scale(0.98); }
.fx-icbtn:active { transform: scale(0.92); }
.ch-top .tab:active { transform: scale(0.98); }
.fx-btn, .fx-icbtn, .ch-top .tab { transition: background .12s, color .12s, border-color .12s, transform .08s, box-shadow .12s; }

/* ─── Pulse: yellow annotation pins ────────────────────────────────── */
@keyframes p-pin-pulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(245,200,66,0.22), 0 4px 12px rgba(0,0,0,0.4); }
  50%      { box-shadow: 0 0 0 8px rgba(245,200,66,0.0),  0 4px 12px rgba(0,0,0,0.4); }
}
.a-keyframe .pin { animation: p-pin-pulse 2.4s ease-in-out infinite; }

/* ─── Tooltip (pure CSS via data-tip) ──────────────────────────────── */
.fx-app [data-tip] { position: relative; }
.fx-app [data-tip]::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px); left: 50%;
  transform: translateX(-50%) translateY(4px);
  padding: 4px 8px; border-radius: 4px;
  background: #050608; color: var(--t);
  font-family: var(--sans); font-size: 11px; font-weight: 500;
  white-space: nowrap;
  border: 1px solid var(--bd2);
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  opacity: 0; pointer-events: none;
  transition: opacity .12s, transform .12s;
  transition-delay: 0s;
  z-index: 1000;
  letter-spacing: -0.005em;
}
.fx-app [data-tip][data-tip-pos="bottom"]::after {
  bottom: auto; top: calc(100% + 6px);
  transform: translateX(-50%) translateY(-4px);
}
.fx-app [data-tip][data-tip-pos="right"]::after {
  bottom: auto; top: 50%; left: calc(100% + 6px);
  transform: translateY(-50%) translateX(-4px);
}
.fx-app [data-tip]:hover::after {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
  transition-delay: .35s;
}
.fx-app [data-tip][data-tip-pos="right"]:hover::after {
  transform: translateY(-50%) translateX(0);
}

/* ─── Command palette ──────────────────────────────────────────────── */
.cp-back {
  position: fixed; inset: 0;
  background: rgba(7,8,11,0.62);
  backdrop-filter: blur(8px);
  display: flex; align-items: flex-start; justify-content: center;
  padding-top: 14vh;
  z-index: 10000;
  animation: p-fade-in 160ms ease-out;
}
.cp-panel {
  width: 640px; max-width: calc(100vw - 60px);
  max-height: 60vh;
  background: var(--panel);
  border: 1px solid var(--bd2);
  border-radius: 11px;
  box-shadow:
    0 30px 80px rgba(0,0,0,0.55),
    0 0 0 1px rgba(139,123,216,0.18);
  display: flex; flex-direction: column;
  overflow: hidden;
}
.cp-input {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--bd);
}
.cp-input .ico { color: var(--muted); display: flex; align-items: center; }
.cp-input input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; font-size: 14.5px; color: var(--t);
  letter-spacing: -0.005em;
}
.cp-input input::placeholder { color: var(--muted); }
.cp-input .esc {
  font-family: var(--mono); font-size: 10px; padding: 2px 6px;
  border-radius: 4px; border: 1px solid var(--bd2);
  color: var(--muted); background: var(--raised);
}

.cp-list { flex: 1; overflow-y: auto; padding: 6px 0; }
.cp-group {
  padding: 8px 18px 4px;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.cp-item {
  display: grid; grid-template-columns: 22px 1fr auto auto;
  gap: 10px; align-items: center;
  padding: 7px 18px; cursor: pointer;
  font-size: 13px; color: var(--t2);
  transition: background .08s;
}
.cp-item:hover { background: var(--hover); }
.cp-item.sel { background: var(--ac-bg); color: var(--t); }
.cp-item .ic { color: var(--muted); display: flex; align-items: center; }
.cp-item.sel .ic { color: var(--ac); }
.cp-item .lab .nm { color: var(--t); font-weight: 500; }
.cp-item .lab .sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
.cp-item .filmdot { width: 7px; height: 7px; border-radius: 50%; }
.cp-item .kbd {
  font-family: var(--mono); font-size: 9.5px; padding: 1px 6px;
  border: 1px solid var(--bd2); border-radius: 3px;
  color: var(--muted); background: var(--bg);
}
.cp-item .badge {
  font-family: var(--mono); font-size: 9.5px;
  padding: 1px 6px; border-radius: 3px;
  background: var(--raised); color: var(--t2);
}
.cp-foot {
  padding: 10px 18px;
  display: flex; align-items: center; justify-content: space-between;
  border-top: 1px solid var(--bd);
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  background: var(--bg);
}
.cp-foot .keys { display: flex; gap: 14px; align-items: center; }
.cp-foot .keys .k b { color: var(--t); font-weight: 500; margin-right: 4px; }

.cp-empty {
  padding: 38px 18px 32px;
  text-align: center; color: var(--muted);
  font-size: 12.5px;
}
.cp-empty .big { font-size: 32px; color: var(--faint); margin-bottom: 8px; }

/* ─── Keyboard help overlay (?) ────────────────────────────────────── */
.kh-back {
  position: fixed; inset: 0;
  background: rgba(7,8,11,0.62);
  backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  padding: 30px;
  z-index: 10000;
  animation: p-fade-in 160ms ease-out;
}
.kh-panel {
  width: 720px; max-width: calc(100vw - 60px);
  max-height: calc(100vh - 60px);
  background: var(--panel);
  border: 1px solid var(--bd2);
  border-radius: 11px;
  box-shadow: 0 30px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(139,123,216,0.18);
  display: flex; flex-direction: column; overflow: hidden;
}
.kh-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 22px; border-bottom: 1px solid var(--bd);
}
.kh-head h2 {
  margin: 0; font-size: 17px; font-weight: 600; color: var(--t);
  letter-spacing: -0.012em;
}
.kh-head h2 .pip {
  font-family: var(--mono); font-size: 10.5px;
  padding: 2px 7px; border-radius: 4px;
  background: var(--ac-bg); color: var(--ac);
  margin-left: 8px; font-weight: 500;
}
.kh-head .close {
  width: 30px; height: 30px; border-radius: 6px;
  background: transparent; border: 1px solid var(--bd2);
  color: var(--t2); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 14px;
}
.kh-head .close:hover { background: var(--hover); color: var(--t); }

.kh-body {
  flex: 1; overflow-y: auto; padding: 18px 22px 22px;
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px 28px;
  align-content: start;
}
.kh-group {
  padding: 4px 0 12px;
}
.kh-group h3 {
  margin: 0 0 10px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.kh-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 6px 0;
  border-top: 1px solid var(--bd);
}
.kh-row:first-of-type { border-top: none; }
.kh-row .desc { font-size: 12.5px; color: var(--t2); }
.kh-row .keys {
  display: flex; align-items: center; gap: 4px;
}
.kh-row .keys .sep { color: var(--faint); font-family: var(--mono); }
.kh-row .keys .k {
  font-family: var(--mono); font-size: 10.5px;
  padding: 2px 7px; border-radius: 4px;
  background: var(--raised); border: 1px solid var(--bd2);
  color: var(--t); font-weight: 500;
  min-width: 20px; text-align: center;
}

/* ─── Toasts ───────────────────────────────────────────────────────── */
.toast-host {
  position: fixed; right: 18px; bottom: 18px;
  display: flex; flex-direction: column-reverse; gap: 8px;
  z-index: 9000;
  pointer-events: none;
}
@keyframes p-toast-in {
  from { opacity: 0; transform: translateY(10px) scale(.96); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes p-toast-out {
  to { opacity: 0; transform: translateY(-4px) scale(.98); }
}
.toast {
  pointer-events: auto;
  min-width: 260px; max-width: 360px;
  background: var(--panel);
  border: 1px solid var(--bd2);
  border-radius: 8px;
  padding: 11px 14px;
  display: flex; align-items: center; gap: 10px;
  font-size: 12.5px; color: var(--t);
  box-shadow: 0 12px 30px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,0,0,0.3);
  animation: p-toast-in 180ms cubic-bezier(.2,.7,.3,1);
  position: relative; overflow: hidden;
}
.toast.exiting { animation: p-toast-out 200ms ease forwards; }
.toast::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--ac);
}
.toast.success::before { background: var(--green); }
.toast.warn::before { background: var(--orange); }
.toast.error::before { background: var(--red); }
.toast .ic {
  width: 22px; height: 22px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 13px; color: var(--ac);
  background: var(--ac-bg); flex-shrink: 0;
}
.toast.success .ic { color: var(--green); background: var(--green-bg); }
.toast.warn .ic { color: var(--orange); background: rgba(245,144,66,0.14); }
.toast .body { flex: 1; }
.toast .body .ttl { font-weight: 500; color: var(--t); }
.toast .body .sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
.toast .body .sub .mono { font-family: var(--mono); color: var(--t2); }
.toast .close {
  background: transparent; border: none; color: var(--muted);
  cursor: pointer; font-family: var(--mono); font-size: 14px;
  padding: 2px 6px; border-radius: 3px;
}
.toast .close:hover { color: var(--t); background: var(--hover); }

/* ─── Hint chip (bottom-left, "Try ⌘K · ?") ─────────────────────────── */
.poke-chip {
  position: fixed; left: 18px; bottom: 18px;
  display: flex; align-items: center; gap: 10px;
  padding: 7px 12px;
  background: var(--panel);
  border: 1px solid var(--bd);
  border-radius: 18px;
  font-size: 11.5px; color: var(--t2);
  box-shadow: 0 4px 16px rgba(0,0,0,0.35);
  cursor: pointer; z-index: 8500;
  transition: border-color .12s, background .12s, transform .12s;
}
.poke-chip:hover { border-color: var(--bd2); background: var(--hover); transform: translateY(-1px); }
.poke-chip .ic {
  width: 18px; height: 18px; border-radius: 4px;
  background: var(--ac-bg); color: var(--ac);
  display: flex; align-items: center; justify-content: center;
}
.poke-chip .kbd {
  font-family: var(--mono); font-size: 10px;
  padding: 1px 5px; border-radius: 3px;
  background: var(--bg); border: 1px solid var(--bd2);
  color: var(--t); font-weight: 500;
}
.poke-chip .kbd + .kbd { margin-left: -3px; }

/* ─── Subtle row hover on .ev-row & .l-issue & .b-card ─────────────── */
.ev-row, .b-card, .c-cp .scenecard, .r-echo {
  transition: background .1s ease, border-color .12s ease, transform .1s ease, box-shadow .12s ease;
}

/* ─── Better scrollbar everywhere (Firefox) ────────────────────────── */
.fx-app *, .ev-app *, .ab-app * { scrollbar-color: ${FX.border2} transparent; scrollbar-width: thin; }
`;

// ─── Toast bus (global) ──────────────────────────────────────────────
const ToastBus = (() => {
  let nextId = 1;
  const listeners = new Set();
  return {
    push(spec) {
      const id = nextId++;
      const t = { id, kind: 'info', duration: 3500, ...spec };
      listeners.forEach(fn => fn({ type: 'add', toast: t }));
      if (t.duration > 0) {
        setTimeout(() => {
          listeners.forEach(fn => fn({ type: 'remove', id }));
        }, t.duration);
      }
      return id;
    },
    remove(id) { listeners.forEach(fn => fn({ type: 'remove', id })); },
    on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  };
})();

function ToastHost() {
  const [toasts, setToasts] = React.useState([]);
  React.useEffect(() => {
    return ToastBus.on(ev => {
      if (ev.type === 'add') setToasts(arr => [...arr, ev.toast]);
      else if (ev.type === 'remove') setToasts(arr => arr.map(t => t.id === ev.id ? {...t, exiting: true} : t));
    });
  }, []);
  // Clean up exiting toasts after animation
  React.useEffect(() => {
    const exiting = toasts.find(t => t.exiting);
    if (exiting) {
      const tm = setTimeout(() => setToasts(arr => arr.filter(t => t.id !== exiting.id)), 220);
      return () => clearTimeout(tm);
    }
  }, [toasts]);

  return (
    <div className="toast-host">
      {toasts.map(t => (
        <div key={t.id} className={'toast ' + (t.kind || 'info') + (t.exiting ? ' exiting' : '')}>
          <div className="ic">
            {t.kind === 'success' ? <I.check /> :
             t.kind === 'warn' ? '!' :
             t.kind === 'error' ? '×' :
             <I.bell />}
          </div>
          <div className="body">
            <div className="ttl">{t.title}</div>
            {t.sub && <div className="sub">{t.sub}</div>}
          </div>
          <button className="close" onClick={() => ToastBus.remove(t.id)}>×</button>
        </div>
      ))}
    </div>
  );
}

// ─── Command palette ─────────────────────────────────────────────────
function CommandPalette({ open, onClose, setTab }) {
  const films = window.FILMS;
  const filmsById = Object.fromEntries(films.map(f => [f.id, f]));
  const filmKey = (id) => ({jeca:'JECA',limite:'LIMT',rio40:'R40G',cangaceiro:'CANG',aruanda:'ARUA',pagador:'PAGD'}[id]);

  const [q, setQ] = React.useState('');
  const [sel, setSel] = React.useState(0);
  const inputRef = React.useRef(null);

  React.useEffect(() => {
    if (open) {
      setQ('');
      setSel(0);
      setTimeout(() => inputRef.current && inputRef.current.focus(), 30);
    }
  }, [open]);

  // Action items
  const ALL = React.useMemo(() => {
    const tabs = [
      { kind:'tab', key:'buscar', label:'Ir para Buscar', sub:'Busca semântica · híbrido + rerank', kbd:'1', ic:<I.search /> },
      { kind:'tab', key:'cenas',  label:'Ir para Cenas',  sub:'Acervo inteiro · 1.588 cenas', kbd:'2', ic:<I.grid /> },
      { kind:'tab', key:'anotar', label:'Ir para Anotar', sub:'Revisão de cena focada', kbd:'3', ic:<I.tag /> },
      { kind:'tab', key:'rimas',  label:'Ir para Rimas visuais', sub:'kNN cross-film · 8 echoes', kbd:'4', ic:<I.rhymes /> },
      { kind:'tab', key:'proc',   label:'Ir para Processamento', sub:'Pipeline ativo · aruanda 78%', kbd:'5', ic:<I.proc /> },
    ];
    const actions = [
      { kind:'action', label:'Iniciar processamento…',   sub:'Abrir seleção de filme + pipeline', ic:<I.proc />, do: () => {
        setTab('proc');
        ToastBus.push({title:'Abrindo Processamento', sub:<>Selecione um filme novo para indexar</>, kind:'info'});
      }},
      { kind:'action', label:'Trocar âncora · Rimas visuais', sub:'Escolher cena âncora', ic:<I.image />, do: () => {
        setTab('rimas');
        ToastBus.push({title:'Pronto para trocar âncora', sub:'Clique numa cena pra defini-la como âncora', kind:'info'});
      }},
      { kind:'action', label:'Copiar timecode da cena atual', sub:'⌘C', ic:<I.link />, kbd:'⌘C', do: () => {
        ToastBus.push({title:'Timecode copiado', sub:<><span style={{fontFamily:'var(--mono)', color: FX.ac}}>00:21:58:08</span> · JECA-111</>, kind:'success'});
      }},
      { kind:'action', label:'Exportar eval set · formato TREC', sub:'eval-2026-04 · 42 queries · 378 julgamentos', ic:<I.download />, kbd:'⌘E', do: () => {
        ToastBus.push({title:'Exportando eval set', sub:<><span className="mono">eval-2026-04.qrels</span> · TREC qrels v1</>, kind:'info'});
      }},
      { kind:'action', label:'Alternar modo cego (Eval)', sub:'Ocultar scores durante grading', ic:<I.settings />, do: () => {
        ToastBus.push({title:'Modo cego ativado', sub:'Scores do modelo agora ocultos no Eval', kind:'success'});
      }},
      { kind:'action', label:'Abrir About', sub:'Atribuições, modelos, créditos', ic:<I.doc />, do: () => {
        ToastBus.push({title:'Abrindo About', sub:'Atribuições e créditos institucionais', kind:'info'});
      }},
      { kind:'action', label:'Compartilhar coleção atual', sub:'Link público + senha opcional', ic:<I.share />, do: () => {
        ToastBus.push({title:'Link copiado', sub:<><span className="mono">mojica.local/s/r7g3k9</span></>, kind:'success'});
      }},
    ];
    const filmItems = films.map(f => ({
      kind:'film', key:f.id, label:f.title, sub:`${f.year} · ${f.director} · ${f.scenes} cenas · ${f.runtime}m`,
      filmColor: window.FX_FILM[f.id], ic:<I.film />, do: () => {
        setTab('cenas');
        ToastBus.push({title:`Filtrando por ${f.title}`, sub:`${f.scenes} cenas · ${f.runtime}m`, kind:'info'});
      }
    }));
    const scenes = window.RESULTS.slice(0, 6).map(r => {
      const f = filmsById[r.film];
      return {
        kind:'scene', key:r.id, label:`${filmKey(r.film)}-${String(r.cena).padStart(3,'0')}`,
        sub: r.desc,
        filmColor: window.FX_FILM[r.film],
        ic:<I.image />,
        do: () => {
          setTab('anotar');
          ToastBus.push({title:`Abrindo cena ${filmKey(r.film)}-${String(r.cena).padStart(3,'0')}`, sub:f.title + ' · ' + r.tc, kind:'info'});
        }
      };
    });
    return { tabs, actions, films: filmItems, scenes };
  }, [films, filmsById, setTab]);

  const ql = q.trim().toLowerCase();
  const match = (item) => !ql ||
    (item.label || '').toLowerCase().includes(ql) ||
    (item.sub || '').toLowerCase().includes(ql);

  const groups = [
    { name: 'Navegar', items: ALL.tabs.filter(match) },
    { name: 'Ações', items: ALL.actions.filter(match) },
    { name: 'Filmes · ' + films.length, items: ALL.films.filter(match) },
    { name: 'Cenas recentes', items: ALL.scenes.filter(match) },
  ].filter(g => g.items.length > 0);

  const flatItems = groups.flatMap(g => g.items);
  const visibleCount = flatItems.length;

  // Keyboard nav
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); setSel(s => Math.min(visibleCount - 1, s + 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setSel(s => Math.max(0, s - 1)); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        const it = flatItems[sel];
        if (it) {
          if (it.kind === 'tab') setTab(it.key);
          else if (it.do) it.do();
          onClose();
        }
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, sel, visibleCount, flatItems, onClose, setTab]);

  if (!open) return null;
  let idx = -1;

  return (
    <div className="cp-back" onClick={onClose}>
      <div className="cp-panel" onClick={(e) => e.stopPropagation()}>
        <div className="cp-input">
          <span className="ico"><I.search /></span>
          <input
            ref={inputRef}
            placeholder="Buscar comandos, filmes, cenas… (digite ‹›› ou › para subcomandos)"
            value={q}
            onChange={(e) => { setQ(e.target.value); setSel(0); }}
          />
          <span className="esc">esc</span>
        </div>
        <div className="cp-list">
          {visibleCount === 0 ? (
            <div className="cp-empty">
              <div className="big">⌕</div>
              <div>Nada encontrado para "<span style={{color: FX.t}}>{q}</span>"</div>
              <div style={{marginTop: 6, fontSize: 11, color: FX.faint}}>Try buscar, anotar, abrir [filme], iniciar processamento…</div>
            </div>
          ) : groups.map(g => (
            <React.Fragment key={g.name}>
              <div className="cp-group">{g.name}</div>
              {g.items.map(it => {
                idx++; const isSel = idx === sel;
                return (
                  <div key={(it.kind || '') + '-' + (it.key || it.label)}
                       className={'cp-item' + (isSel ? ' sel' : '')}
                       onClick={() => {
                         if (it.kind === 'tab') setTab(it.key);
                         else if (it.do) it.do();
                         onClose();
                       }}
                       onMouseEnter={() => setSel(idx)}>
                    <span className="ic">
                      {it.filmColor
                        ? <span className="filmdot" style={{background: it.filmColor}}></span>
                        : it.ic}
                    </span>
                    <div className="lab">
                      <span className="nm">{it.label}</span>
                      <div className="sub">{it.sub}</div>
                    </div>
                    {it.kind === 'film' && <span className="badge">{filmKey(it.key)}</span>}
                    {it.kind === 'scene' && <span className="badge">cena</span>}
                    {it.kbd && <span className="kbd">{it.kbd}</span>}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
        <div className="cp-foot">
          <div className="keys">
            <span className="k"><b>↑↓</b> nav</span>
            <span className="k"><b>⏎</b> select</span>
            <span className="k"><b>esc</b> close</span>
          </div>
          <div className="keys">
            <span className="k">{visibleCount} resultados</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Keyboard help overlay (?) ───────────────────────────────────────
function HelpOverlay({ open, onClose }) {
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape' || e.key === '?') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="kh-back" onClick={onClose}>
      <div className="kh-panel" onClick={(e) => e.stopPropagation()}>
        <div className="kh-head">
          <h2>Atalhos de teclado <span className="pip">?</span></h2>
          <button className="close" onClick={onClose}>×</button>
        </div>
        <div className="kh-body">
          <div className="kh-group">
            <h3>Navegação</h3>
            <Row d="Buscar"                 ks={['1']} />
            <Row d="Cenas"                  ks={['2']} />
            <Row d="Anotar"                 ks={['3']} />
            <Row d="Rimas visuais"          ks={['4']} />
            <Row d="Processamento"          ks={['5']} />
            <Row d="Item seguinte · anterior" ks={['j','k']} sep="·" />
            <Row d="Selecionar"             ks={['⏎']} />
            <Row d="Voltar · fechar"        ks={['esc']} />
          </div>

          <div className="kh-group">
            <h3>Universal</h3>
            <Row d="Paleta de comandos"     ks={['⌘','K']} sep="+" />
            <Row d="Esta ajuda"             ks={['?']} />
            <Row d="Buscar global"          ks={['⌘','F']} sep="+" />
            <Row d="Voltar"                 ks={['⌘','[']} sep="+" />
            <Row d="Recarregar índice"      ks={['⌘','R']} sep="+" />
            <Row d="Exportar"               ks={['⌘','E']} sep="+" />
          </div>

          <div className="kh-group">
            <h3>Anotar · revisão</h3>
            <Row d="Reproduzir · pausar"   ks={['espaço']} />
            <Row d="Voltar · avançar 5s"   ks={['←','→']} sep="·" />
            <Row d="Fixar comentário no quadro" ks={['P']} />
            <Row d="Resolver comentário"   ks={['⇧','R']} sep="+" />
            <Row d="Cena anterior · próxima" ks={['⇧','J']} sep="+" />
          </div>

          <div className="kh-group">
            <h3>Eval · grading</h3>
            <Row d="Grade · 0 / 1 / 2 / 3" ks={['0','1','2','3']} sep="·" />
            <Row d="Skip · não julgável"   ks={['S']} />
            <Row d="Salvar query e avançar" ks={['⌘','⏎']} sep="+" />
            <Row d="Modo cego (toggle)"    ks={['⇧','B']} sep="+" />
            <Row d="Pular query"           ks={['⇧','S']} sep="+" />
          </div>

          <div className="kh-group" style={{gridColumn: '1 / 3'}}>
            <h3>Cenas + Buscar</h3>
            <Row d="Trocar âncora · Rimas visuais"  ks={['⌥','R']} sep="+" />
            <Row d="Copiar timecode"                ks={['⌘','C']} sep="+" />
            <Row d="Adicionar tag"                  ks={['A']} />
            <Row d="Alternar modo de exibição (grade · lista · compacto)" ks={['G']} />
            <Row d="Selecionar tudo · limpar"       ks={['⌘','A']} sep="+" />
            <Row d="Filtrar"                        ks={['⌘','⇧','F']} sep="+" />
          </div>
        </div>
      </div>
    </div>
  );
}
function Row({ d, ks, sep = '' }) {
  return (
    <div className="kh-row">
      <span className="desc">{d}</span>
      <span className="keys">
        {ks.map((k, i) => (
          <React.Fragment key={i}>
            {i > 0 && sep && <span className="sep">{sep}</span>}
            <span className="k">{k}</span>
          </React.Fragment>
        ))}
      </span>
    </div>
  );
}

// ─── Hint chip ───────────────────────────────────────────────────────
function PokeChip({ openPalette, openHelp }) {
  return (
    <div className="poke-chip" data-tip="Atalhos disponíveis">
      <span className="ic">⌃</span>
      <span>Try</span>
      <span className="kbd" onClick={openPalette}>⌘ K</span>
      <span style={{color: FX.faint}}>·</span>
      <span className="kbd" onClick={openHelp}>?</span>
    </div>
  );
}

// ─── Top-level wrapper component ─────────────────────────────────────
function PolishLayer({ setTab }) {
  const [cpOpen, setCpOpen] = React.useState(false);
  const [helpOpen, setHelpOpen] = React.useState(false);

  React.useEffect(() => {
    const onKey = (e) => {
      // ⌘K or Ctrl+K — open command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setHelpOpen(false);
        setCpOpen(o => !o);
      }
      // ? — help (only when not typing)
      if (e.key === '?' && !(e.metaKey || e.ctrlKey || e.altKey)) {
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;
        e.preventDefault();
        setCpOpen(false);
        setHelpOpen(o => !o);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  React.useEffect(() => {
    if (!document.getElementById('polish-css')) {
      const s = document.createElement('style');
      s.id = 'polish-css'; s.textContent = POLISH_CSS;
      document.head.appendChild(s);
    }
  }, []);

  return (
    <>
      <ToastHost />
      <CommandPalette open={cpOpen} onClose={() => setCpOpen(false)} setTab={setTab} />
      <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
      <PokeChip openPalette={() => setCpOpen(true)} openHelp={() => setHelpOpen(true)} />
    </>
  );
}

window.PolishLayer = PolishLayer;
window.ToastBus = ToastBus;
window.POLISH_CSS = POLISH_CSS;
