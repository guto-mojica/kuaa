// Mojica · Frame.io branch · App with tab routing

function MojicaApp() {
  const [tab, setTab] = React.useState('buscar');
  const [selected, setSelected] = React.useState(0);

  // Inject all scoped CSS once
  React.useEffect(() => {
    if (!document.getElementById('mfx-css')) {
      const s = document.createElement('style');
      s.id = 'mfx-css';
      s.textContent = [
        window.FX_CSS,
        window.CH_CSS,
        window.BUSCAR_CSS,
        window.CENAS_CSS,
        window.ANOTAR_CSS,
        window.RIMAS_CSS,
        window.PROC_CSS,
      ].join('\n');
      document.head.appendChild(s);
    }
  }, []);

  // Keyboard nav: j/k or ↑/↓ for selection in screens with lists,
  // 1/2/3/4/5 to switch tabs.
  React.useEffect(() => {
    const onKey = (e) => {
      if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); setSelected(s => s + 1); }
      else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); setSelected(s => Math.max(0, s - 1)); }
      else if (e.key === '1') setTab('buscar');
      else if (e.key === '2') setTab('cenas');
      else if (e.key === '3') setTab('anotar');
      else if (e.key === '4') setTab('rimas');
      else if (e.key === '5') setTab('proc');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const films = window.FILMS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));

  // Per-tab info
  const tabInfo = {
    buscar: {
      breadcrumb: [{label: 'Acervo'}, {label: 'Buscar', cur: true}, {label: 'V3 · index', ver: true}],
      compact: false,
      // selected scene is from window.RESULTS by index
      sceneSel: window.RESULTS[Math.min(selected, window.RESULTS.length - 1)] || window.RESULTS[0],
    },
    cenas: {
      breadcrumb: [{label: 'Acervo'}, {label: 'Cenas — Acervo inteiro', cur: true}],
      compact: false,
      sceneSel: window.RESULTS[0],
    },
    anotar: {
      breadcrumb: [
        {label: 'Acervo'},
        {label: byId[window.RESULTS[Math.min(selected, window.RESULTS.length - 1)].film].title},
        {label: 'cena ' + String(window.RESULTS[Math.min(selected, window.RESULTS.length - 1)].cena).padStart(3,'0'), cur: true},
        {label: 'V3', ver: true},
      ],
      compact: true,
      sceneSel: window.RESULTS[Math.min(selected, window.RESULTS.length - 1)],
    },
    rimas: {
      breadcrumb: [{label: 'Acervo'}, {label: 'Rimas visuais', cur: true}, {label: 'jeca · cena 003 · âncora', ver: true}],
      compact: false,
      sceneSel: { film: 'jeca', kf: 'keyframes/kf-03-horse.jpg', tc: '00:01:57:18', cena: 3 },
    },
    proc: {
      breadcrumb: [{label: 'Acervo'}, {label: 'Processamento', cur: true}, {label: 'aruanda · 78%', ver: true}],
      compact: false,
      sceneSel: null,
    },
  };

  const info = tabInfo[tab];
  const selectedFilm = info.sceneSel ? byId[info.sceneSel.film] : null;
  const bodyClass = 'ch-body with-right' + (info.compact ? ' compact-lp' : '');

  return (
    <div className="fx-app">
      <TopBar tab={tab} setTab={setTab} breadcrumb={info.breadcrumb} />
      <div className={bodyClass}>
        <IconRail tab={tab} setTab={setTab} />
        {!info.compact && (
          <LeftPane selectedFilm={selectedFilm} selectedScene={info.sceneSel} />
        )}

        {tab === 'buscar' && <ScreenBuscar selected={selected % window.RESULTS.length} setSelected={(i) => setSelected(i)} />}
        {tab === 'cenas'  && <ScreenCenas  selected={selected} setSelected={(i) => setSelected(i)} />}
        {tab === 'anotar' && <ScreenAnotar selected={selected % window.RESULTS.length} setSelected={(i) => setSelected(i)} />}
        {tab === 'rimas'  && <ScreenRimas  selected={selected} setSelected={(i) => setSelected(i)} />}
        {tab === 'proc'   && <ScreenProc />}
      </div>
      <PolishLayer setTab={setTab} />
    </div>
  );
}

window.MojicaApp = MojicaApp;
