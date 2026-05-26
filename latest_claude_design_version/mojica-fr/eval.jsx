// Mojica · Eval set builder (admin/internal)
// "Measuring instrument, not a feature" — dense rows, tabular figures,
// live P@K/NDCG metrics, inter-annotator agreement.

const EV_CSS = `
.ev-app, .ev-app * { box-sizing: border-box; }
.ev-app *::-webkit-scrollbar { width: 8px; height: 8px; }
.ev-app *::-webkit-scrollbar-thumb { background: ${FX.border2}; border-radius: 4px; }
.ev-app *::-webkit-scrollbar-track { background: transparent; }

.ev-app {
  display: grid;
  grid-template-rows: 24px 50px 1fr 30px;
  height: 100vh; width: 100vw;
  background: var(--bg); color: var(--t);
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  letter-spacing: -0.003em;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'ss01' on, 'tnum' on;
  overflow: hidden;
}

/* ───── ADMIN STRIP ─────────────────────────────────────────────────── */
.ev-admin {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px; background: rgba(245,144,66,0.08);
  border-bottom: 1px solid rgba(245,144,66,0.2);
  font-size: 10.5px; color: var(--orange);
}
.ev-admin .l {
  display: flex; align-items: center; gap: 14px;
  font-family: var(--mono); letter-spacing: 0.16em; text-transform: uppercase; font-weight: 500;
}
.ev-admin .l .badge {
  background: var(--orange); color: #0E1014; padding: 1px 7px; border-radius: 0;
  font-weight: 700;
}
.ev-admin .r {
  font-family: var(--mono); font-size: 10px; color: var(--orange);
  display: flex; align-items: center; gap: 18px;
  letter-spacing: 0.04em;
}
.ev-admin .r b { color: #FFB075; font-weight: 600; }

/* ───── HEADER ──────────────────────────────────────────────────────── */
.ev-top {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px; background: var(--panel);
  border-bottom: 1px solid var(--bd);
}
.ev-top .l { display: flex; align-items: center; gap: 14px; }
.ev-top .brand { display: flex; align-items: center; gap: 9px; }
.ev-top .brand .n {
  font-weight: 600; font-size: 14.5px; color: var(--t); letter-spacing: -0.01em;
}
.ev-top .brand .sub {
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
  padding-left: 11px; border-left: 1px solid var(--bd2); margin-left: 4px;
}
.ev-top .div { width: 1px; height: 22px; background: var(--bd); }
.ev-top .crumb {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; color: var(--t2);
}
.ev-top .crumb .seg { padding: 2px 4px; }
.ev-top .crumb .seg.cur { color: var(--t); font-weight: 500; }
.ev-top .crumb .sep { color: var(--faint); }
.ev-top .crumb .runid {
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  padding: 2px 7px; border-radius: 4px; background: var(--raised); border: 1px solid var(--bd);
}
.ev-top .session-counter {
  display: flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 11px;
}
.ev-top .session-counter .n { color: var(--t); font-weight: 600; font-variant-numeric: tabular-nums; }
.ev-top .session-counter .k { color: var(--muted); }
.ev-top .session-counter .bar {
  width: 80px; height: 4px; background: var(--bd); border-radius: 2px; position: relative;
}
.ev-top .session-counter .bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac); border-radius: 2px;
}
.ev-top .r { display: flex; align-items: center; gap: 6px; }
.ev-top .me {
  display: flex; align-items: center; gap: 7px; padding: 4px 10px 4px 5px;
  background: var(--raised); border: 1px solid var(--bd); border-radius: 14px;
}
.ev-top .me .av {
  width: 22px; height: 22px; border-radius: 50%;
  background: linear-gradient(135deg, var(--ac), var(--pink));
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 10px; font-weight: 700; color: #fff;
}
.ev-top .me .name { font-size: 12px; color: var(--t); }
.ev-top .me .role { font-family: var(--mono); font-size: 10px; color: var(--ac); }

/* ───── BODY ────────────────────────────────────────────────────────── */
.ev-body { display: grid; grid-template-columns: 286px 1fr 388px; overflow: hidden; min-height: 0; }

/* LEFT PANE — queries queue */
.ev-lp {
  border-right: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.ev-lp .head {
  padding: 14px 14px 8px;
  display: flex; align-items: baseline; justify-content: space-between;
  font-size: 11px; font-weight: 500; color: var(--muted);
  letter-spacing: 0.04em;
}
.ev-lp .head .v { font-family: var(--mono); color: var(--t); font-size: 11px; font-weight: 500; }
.ev-lp .filter {
  margin: 0 12px 8px; padding: 6px 10px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 6px;
  display: flex; align-items: center; gap: 8px;
  font-size: 12px;
}
.ev-lp .filter input { flex: 1; background: transparent; border: none; outline: none; font: inherit; color: var(--t); }
.ev-lp .filter input::placeholder { color: var(--muted); }
.ev-lp .filter .ico { color: var(--muted); }
.ev-lp .lpf-tabs {
  display: flex; padding: 0 12px 8px; gap: 4px;
}
.ev-lp .lpf-tabs .t {
  flex: 1; text-align: center; padding: 4px 8px; border-radius: 4px;
  font-size: 11px; color: var(--muted); cursor: pointer; font-weight: 500;
  background: var(--bg); border: 1px solid var(--bd);
  display: flex; align-items: center; justify-content: center; gap: 5px;
}
.ev-lp .lpf-tabs .t.on { color: var(--ac); border-color: var(--ac-dim); background: var(--ac-bg); }
.ev-lp .lpf-tabs .t .ct {
  font-family: var(--mono); font-size: 10px; padding: 0 5px;
  background: var(--raised); border-radius: 8px;
}
.ev-lp .lpf-tabs .t.on .ct { background: var(--ac-dim); color: var(--t); }

.ev-q-list { flex: 1; overflow-y: auto; padding: 4px 0 8px; }
.ev-q {
  display: grid; grid-template-columns: 18px 1fr;
  gap: 9px; align-items: start;
  padding: 8px 14px; cursor: pointer; position: relative;
  border-bottom: 1px solid rgba(38,44,54,0.5);
}
.ev-q:hover { background: var(--hover); }
.ev-q.cur { background: var(--ac-bg); }
.ev-q.cur::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--ac);
}
.ev-q .stat {
  width: 14px; height: 14px; border-radius: 50%;
  border: 1.5px solid var(--bd2);
  display: flex; align-items: center; justify-content: center;
  margin-top: 2px;
  position: relative;
}
.ev-q .stat.done {
  background: var(--green); border-color: var(--green);
  color: #0E1014;
}
.ev-q.cur .stat { border-color: var(--ac); }
.ev-q.cur .stat::before {
  content: ''; position: absolute; inset: 2px; border-radius: 50%;
  background: var(--ac);
}
.ev-q .body { min-width: 0; }
.ev-q .qrow {
  display: flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: 10px; color: var(--muted);
}
.ev-q .qrow .id { color: var(--t); font-weight: 500; }
.ev-q .qrow .lang { color: var(--ac); }
.ev-q .qrow .src { color: var(--muted); font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.06em; }
.ev-q.cur .qrow .id { color: var(--ac); }
.ev-q .text {
  font-size: 12.5px; color: var(--t2); line-height: 1.4; margin-top: 3px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  text-wrap: pretty;
}
.ev-q.cur .text { color: var(--t); }
.ev-q .progress {
  display: flex; align-items: center; gap: 5px; margin-top: 6px;
  font-family: var(--mono); font-size: 10px; color: var(--muted);
}
.ev-q .progress .pips {
  display: flex; gap: 2px;
}
.ev-q .progress .pips .pp {
  width: 6px; height: 6px; border-radius: 1px;
}
.ev-q .progress .pips .pp.g0 { background: var(--red); }
.ev-q .progress .pips .pp.g1 { background: var(--orange); }
.ev-q .progress .pips .pp.g2 { background: var(--green); }
.ev-q .progress .pips .pp.g3 { background: var(--ac); }
.ev-q .progress .pips .pp.gp { background: var(--bd2); }

.ev-lp .controls {
  border-top: 1px solid var(--bd);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 7px;
}
.ev-lp .controls .toggle {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 12px; color: var(--t2); cursor: pointer;
  padding: 4px 0;
}
.ev-lp .controls .toggle .v {
  font-family: var(--mono); font-size: 10.5px;
  display: inline-flex; align-items: center; gap: 5px;
}
.ev-lp .controls .toggle .swt {
  width: 26px; height: 14px; background: var(--bd2); border-radius: 8px;
  position: relative; transition: background .15s;
}
.ev-lp .controls .toggle .swt::before {
  content: ''; position: absolute; left: 2px; top: 2px;
  width: 10px; height: 10px; border-radius: 50%; background: var(--muted);
  transition: left .15s, background .15s;
}
.ev-lp .controls .toggle.on .swt { background: var(--ac); }
.ev-lp .controls .toggle.on .swt::before { left: 14px; background: #fff; }

/* CENTER PANE — current query + candidates */
.ev-cp { display: flex; flex-direction: column; min-width: 0; overflow: hidden; background: var(--bg); }

.ev-q-card {
  padding: 16px 24px;
  border-bottom: 1px solid var(--bd);
}
.ev-q-card .ll {
  display: flex; align-items: center; gap: 12px;
  font-family: var(--mono); font-size: 10.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 8px;
}
.ev-q-card .ll .qid { color: var(--ac); font-weight: 600; }
.ev-q-card .ll .src {
  padding: 1px 7px; background: var(--raised); border-radius: 3px; color: var(--t2);
  letter-spacing: 0.04em;
}
.ev-q-card .ll .ann {
  display: flex; align-items: center; gap: 4px;
  color: var(--muted); margin-left: auto;
}
.ev-q-card h1 {
  margin: 0; font-size: 22px; font-weight: 600; color: var(--t);
  letter-spacing: -0.014em; line-height: 1.25;
  font-family: var(--mono);
}
.ev-q-card h1 .kind { color: var(--muted); font-family: var(--mono); font-size: 14px; font-weight: 400; margin-right: 6px; }
.ev-q-card .meta {
  display: flex; align-items: center; gap: 18px; margin-top: 8px;
  font-size: 11.5px; color: var(--muted);
}
.ev-q-card .meta b { color: var(--t2); font-weight: 500; }
.ev-q-card .meta .pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 8px; border-radius: 9px; background: var(--raised);
  font-family: var(--mono); font-size: 10.5px;
}
.ev-q-card .meta .pill .dot { width: 6px; height: 6px; border-radius: 50%; }

/* Sub-control row above results */
.ev-cp .subbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 24px;
  border-bottom: 1px solid var(--bd); background: var(--bg);
}
.ev-cp .subbar .l {
  display: flex; align-items: center; gap: 14px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}
.ev-cp .subbar .l b { color: var(--t); font-weight: 500; }
.ev-cp .subbar .r { display: flex; align-items: center; gap: 4px; }

/* GRADE BUTTONS LEGEND */
.ev-cp .legend {
  display: flex; align-items: center; gap: 4px;
  padding: 8px 24px 6px;
  font-family: var(--mono); font-size: 10px;
  color: var(--muted); letter-spacing: 0.04em;
}
.ev-cp .legend .lab { margin-right: 4px; }
.ev-cp .legend .chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 7px; border-radius: 3px;
  font-weight: 500;
}
.ev-cp .legend .chip.g0 { background: var(--red-bg); color: var(--red); }
.ev-cp .legend .chip.g1 { background: var(--orange-bg); color: var(--orange); }
.ev-cp .legend .chip.g2 { background: var(--green-bg); color: var(--green); }
.ev-cp .legend .chip.g3 { background: var(--ac-bg); color: var(--ac); }
.ev-cp .legend .chip.sk { background: var(--raised); color: var(--muted); }

/* ROWS */
.ev-rows { flex: 1; overflow-y: auto; padding: 0 12px 18px; }
.ev-row {
  display: grid;
  grid-template-columns: 38px 192px 88px 1fr 78px;
  gap: 12px; align-items: center;
  padding: 10px 12px; cursor: pointer; position: relative;
  border-bottom: 1px solid var(--bd);
  transition: background .08s;
}
.ev-row:hover { background: var(--hover); }
.ev-row.cur { background: var(--ac-bg); }
.ev-row.cur::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
  background: var(--ac);
}
.ev-row.graded::after {
  content: ''; position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
  width: 0; height: 0; pointer-events: none;
}

.ev-row .rank {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  font-family: var(--mono); font-size: 13px; color: var(--t);
  font-weight: 600; font-variant-numeric: tabular-nums;
}
.ev-row .rank .rk {
  font-size: 10px; color: var(--muted); font-weight: 400;
  letter-spacing: 0.04em;
}
.ev-row .rank .rk b { color: var(--t2); font-weight: 500; }

.ev-row .grades {
  display: grid; grid-template-columns: repeat(4, 1fr) 28px;
  gap: 3px;
  background: var(--bg); border: 1px solid var(--bd2); border-radius: 5px;
  padding: 2px;
}
.ev-row .grades .gb {
  height: 28px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 12px; font-weight: 600;
  color: var(--muted); cursor: pointer; border-radius: 3px;
  border: none; background: transparent;
  transition: background .12s, color .12s;
}
.ev-row .grades .gb.k {
  font-size: 11px; color: var(--faint);
}
.ev-row .grades .gb:hover { background: var(--hover); color: var(--t); }
.ev-row .grades .gb.on0 { background: var(--red); color: #fff; }
.ev-row .grades .gb.on1 { background: var(--orange); color: #0E1014; }
.ev-row .grades .gb.on2 { background: var(--green); color: #0E1014; }
.ev-row .grades .gb.on3 { background: var(--ac); color: #fff; }
.ev-row .grades .gb.ons { background: var(--raised); color: var(--t); }
.ev-row .grades .gb.dim { opacity: 0.3; }

.ev-row .thumb {
  width: 88px; height: 66px; border-radius: 4px;
  background-size: cover; background-position: center;
  filter: contrast(1.04) brightness(0.96);
  border: 1px solid var(--bd);
}

.ev-row .body { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.ev-row .body .head1 {
  display: flex; align-items: center; gap: 8px;
  font-size: 12.5px;
}
.ev-row .body .head1 .id { font-family: var(--mono); color: var(--muted); font-size: 11px; }
.ev-row .body .head1 .filmpill {
  display: inline-flex; align-items: center; gap: 5px;
}
.ev-row .body .head1 .filmpill .dot { width: 7px; height: 7px; border-radius: 50%; }
.ev-row .body .head1 .filmpill .nm { color: var(--t); font-weight: 500; font-size: 12.5px; }
.ev-row .body .head1 .filmpill .yr { font-family: var(--mono); color: var(--muted); font-size: 10.5px; }
.ev-row .body .head1 .tc { font-family: var(--mono); color: var(--muted); font-size: 10.5px; margin-left: auto; }
.ev-row .body .desc {
  font-size: 12.5px; color: var(--t2); line-height: 1.45;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ev-row.cur .body .desc { color: var(--t); }
.ev-row .body .tags {
  display: flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: 10px; color: var(--faint);
  white-space: nowrap; overflow: hidden;
}
.ev-row .body .tags .t { padding: 0 5px; }

.ev-row .right {
  display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
}
.ev-row .right .score {
  font-family: var(--mono); font-size: 12px; color: var(--t);
  font-variant-numeric: tabular-nums; font-weight: 600;
}
.ev-row .right .score.blind {
  color: var(--faint); letter-spacing: 1px;
}
.ev-row .right .scbar {
  width: 60px; height: 3px; background: var(--bd); border-radius: 2px; position: relative;
}
.ev-row .right .scbar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac); border-radius: 2px;
}
.ev-row .right .scbar.blind::before { background: var(--faint); }
.ev-row .right .src {
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  letter-spacing: 0.04em;
}

/* note row when expanded */
.ev-row.expanded .note-row { display: flex; }
.ev-row .note-row {
  display: none;
  grid-column: 2 / 6; margin-top: 8px;
  align-items: center; gap: 8px;
}
.ev-row .note-row input {
  flex: 1; background: var(--bg); border: 1px solid var(--bd);
  padding: 6px 9px; border-radius: 4px; font: inherit; font-size: 12px;
  color: var(--t); outline: none;
}
.ev-row .note-row input:focus { border-color: var(--ac); }

/* RIGHT PANE — metrics */
.ev-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display: flex; flex-direction: column; overflow: hidden;
}
.ev-rp .head {
  padding: 13px 16px; border-bottom: 1px solid var(--bd);
  display: flex; align-items: center; justify-content: space-between;
}
.ev-rp .head h3 {
  margin: 0; font-size: 12px; font-weight: 600; color: var(--t);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.ev-rp .head .acts { display: flex; gap: 2px; color: var(--muted); }

.ev-rp .inner { padding: 14px 16px 16px; overflow-y: auto; flex: 1; }

/* Big metric cards row */
.ev-bigmets {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  margin-bottom: 14px;
}
.ev-met {
  padding: 10px 12px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
  display: flex; flex-direction: column; gap: 4px;
}
.ev-met .lab {
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.ev-met .val {
  display: flex; align-items: baseline; gap: 7px;
  font-family: var(--mono); font-variant-numeric: tabular-nums;
}
.ev-met .val .n { font-size: 22px; font-weight: 700; color: var(--t); letter-spacing: -0.01em; }
.ev-met .val .d { font-size: 11px; color: var(--muted); }
.ev-met .val .delta { font-size: 10px; padding: 0 4px; border-radius: 3px; margin-left: auto; }
.ev-met .val .delta.up { background: var(--green-bg); color: var(--green); }
.ev-met .val .delta.dn { background: var(--red-bg); color: var(--red); }
.ev-met .bar {
  height: 3px; background: var(--bd); border-radius: 2px; position: relative; margin-top: 3px;
}
.ev-met .bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac); border-radius: 2px;
}
.ev-met.fused .bar::before { background: var(--ac2); }
.ev-met.warn .val .n { color: var(--orange); }

/* Section heads */
.ev-sect {
  display: flex; align-items: baseline; justify-content: space-between;
  margin: 16px 0 8px;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.ev-sect .v { color: var(--t); font-family: var(--mono); }
.ev-sect a { color: var(--ac); cursor: pointer; font-family: var(--mono); text-transform: none; letter-spacing: 0.04em; font-size: 10.5px; }

/* Grade histogram */
.ev-hist {
  padding: 10px 12px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
}
.ev-hist .row {
  display: grid; grid-template-columns: 30px 1fr 24px;
  align-items: center; gap: 8px; margin-bottom: 5px;
  font-family: var(--mono); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.ev-hist .row .g0 { color: var(--red); }
.ev-hist .row .g1 { color: var(--orange); }
.ev-hist .row .g2 { color: var(--green); }
.ev-hist .row .g3 { color: var(--ac); }
.ev-hist .row .gs { color: var(--muted); }
.ev-hist .row .track {
  height: 12px; background: var(--bd); border-radius: 2px; position: relative; overflow: hidden;
}
.ev-hist .row .track::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: var(--p);
}
.ev-hist .row.g0 .track::before { background: var(--red); }
.ev-hist .row.g1 .track::before { background: var(--orange); }
.ev-hist .row.g2 .track::before { background: var(--green); }
.ev-hist .row.g3 .track::before { background: var(--ac); }
.ev-hist .row.gs .track::before { background: var(--muted); }
.ev-hist .row .ct { color: var(--t); text-align: right; }

/* Inter-annotator agreement panel */
.ev-iaa {
  padding: 12px 12px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
}
.ev-iaa .h {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 10px;
  font-size: 12px;
}
.ev-iaa .h .l {
  display: flex; align-items: center; gap: 7px; color: var(--t2); font-weight: 500;
}
.ev-iaa .h .l .av {
  width: 22px; height: 22px; border-radius: 50%;
  background: linear-gradient(135deg, #5CCB91, #2E8A5D); color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 9.5px; font-weight: 700;
}
.ev-iaa .h .r {
  font-family: var(--mono); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.ev-iaa .h .r b { color: var(--green); font-weight: 600; }
.ev-iaa .kappa {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
  padding: 8px 10px; background: var(--panel); border-radius: 5px;
}
.ev-iaa .kappa .label {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  letter-spacing: 0.06em; text-transform: uppercase;
}
.ev-iaa .kappa .val {
  font-family: var(--mono); font-size: 16px; color: var(--green);
  font-weight: 700; font-variant-numeric: tabular-nums;
}
.ev-iaa .kappa .qual {
  font-size: 11px; color: var(--green); margin-left: auto;
}
.ev-iaa .matrix {
  display: grid; grid-template-columns: 36px repeat(5, 1fr); gap: 2px;
  font-family: var(--mono); font-size: 10px;
  font-variant-numeric: tabular-nums;
}
.ev-iaa .matrix .h-cell {
  text-align: center; color: var(--muted); padding: 3px 0;
}
.ev-iaa .matrix .c {
  background: var(--panel); padding: 4px 0;
  text-align: center; border-radius: 2px;
  color: var(--muted);
}
.ev-iaa .matrix .c.hi { background: var(--ac); color: #fff; }
.ev-iaa .matrix .c.mid { background: var(--ac-bg); color: var(--ac); }
.ev-iaa .matrix .c.diag { font-weight: 700; }

/* Session stats card */
.ev-sess {
  padding: 12px; background: var(--bg);
  border: 1px solid var(--bd); border-radius: 7px;
}
.ev-sess .row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 0; font-size: 11.5px;
}
.ev-sess .row + .row { border-top: 1px solid var(--bd); }
.ev-sess .row .k { color: var(--muted); }
.ev-sess .row .v {
  font-family: var(--mono); color: var(--t); font-variant-numeric: tabular-nums;
  font-weight: 500;
}
.ev-sess .row .v.acc { color: var(--ac); }
.ev-sess .row .v.warn { color: var(--orange); }

/* Big save button */
.ev-saverow {
  margin-top: 16px;
  display: flex; flex-direction: column; gap: 6px;
}
.ev-save {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 14px; background: var(--ac); color: #fff;
  border: none; border-radius: 6px; cursor: pointer;
  font: inherit; font-size: 13px; font-weight: 600;
}
.ev-save:hover { background: var(--ac2); }
.ev-save .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 5px;
  background: rgba(0,0,0,0.2); border-radius: 3px;
}
.ev-skip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px; background: transparent; color: var(--t2);
  border: 1px solid var(--bd2); border-radius: 6px; cursor: pointer;
  font: inherit; font-size: 12px;
}
.ev-skip:hover { border-color: var(--bd3); color: var(--t); }

/* BOTTOM STATUS */
.ev-bot {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px;
  border-top: 1px solid var(--bd); background: var(--panel);
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
.ev-bot .mode {
  background: var(--ac); color: #fff;
  padding: 0 8px; height: 18px;
  display: inline-flex; align-items: center; margin-right: 14px;
  font-weight: 600; letter-spacing: 0.08em; font-size: 9.5px;
  text-transform: uppercase;
}
.ev-bot .keys { display: flex; align-items: center; gap: 14px; }
.ev-bot .keys .k b { color: var(--ac); font-weight: 500; margin-right: 4px; }
.ev-bot .keys .k { color: var(--t2); }
.ev-bot .r { display: flex; align-items: center; gap: 16px; }
.ev-bot .r b { color: var(--t); font-weight: 500; font-variant-numeric: tabular-nums; }
`;

// ─── DATA ──────────────────────────────────────────────────────────────
// Mock evaluation queue: 22 queries in this run; 9 done so far.
const EV_QUEUES = [
  { id:'Q-042', lang:'pt', src:'manual',   text:'duas pessoas conversando ao ar livre', grades:[2,3,1,3,2,1,0,2,1], done:9, cur:true },
  { id:'Q-043', lang:'pt', src:'manual',   text:'cena noturna com fogo' },
  { id:'Q-044', lang:'en', src:'auto',     text:'a rider on horseback at the edge of a field' },
  { id:'Q-045', lang:'pt', src:'manual',   text:'cartela de título inicial' },
  { id:'Q-046', lang:'pt', src:'shadow',   text:'mulher segurando objeto em ambiente rural' },
  { id:'Q-047', lang:'en', src:'auto',     text:'two men in checkered shirts in conversation' },
  { id:'Q-048', lang:'pt', src:'manual',   text:'paisagem montanhosa ao fundo de uma cena' },
  { id:'Q-049', lang:'pt', src:'shadow',   text:'mãe e criança em interior modesto' },
  { id:'Q-050', lang:'en', src:'cluster',  text:'wagon in a rural field under a cloudy sky' },
  { id:'Q-051', lang:'pt', src:'auto',     text:'animal de carga em primeiro plano' },
  { id:'Q-052', lang:'pt', src:'manual',   text:'reflexo em água parada' },
];

// Done queries (for the "Concluídas" list — we'll show stat dots)
const EV_DONE = [
  { id:'Q-033', text:'cavalo cruzando o quadro da esquerda para direita', grades:[3,3,2,1,1,2,0,1,0], done:9 },
  { id:'Q-034', text:'plano fechado em rosto de mulher iluminado de lado', grades:[3,2,2,3,1,2,0,1,0], done:9 },
  { id:'Q-035', text:'interior com luz baixa e vela', grades:[2,3,2,1,0,2,1,1,0], done:9 },
  { id:'Q-036', text:'silhueta contra o céu', grades:[3,2,2,2,1,1,0,0,0], done:9 },
  { id:'Q-037', text:'cena de procissão religiosa', grades:[3,2,2,1,1,1,0,0,0], done:9 },
  { id:'Q-038', text:'closeup de mãos trabalhando', grades:[2,3,2,1,2,1,0,0,0], done:9 },
  { id:'Q-039', text:'paisagem árida com vegetação rasteira', grades:[3,3,3,2,1,1,1,0,0], done:9 },
  { id:'Q-040', text:'pano de fundo com céu denso', grades:[2,2,2,1,1,1,0,0,0], done:9 },
  { id:'Q-041', text:'figura de costas em direção ao horizonte', grades:[3,2,3,2,2,1,1,0,0], done:9 },
];

const GRADE_LABELS = {
  0: 'Irrelevante',
  1: 'Vagamente relacionado',
  2: 'Relevante',
  3: 'Altamente relevante',
  s: 'Não julgável',
};

function ScreenEval() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));
  const filmKey = (id) => ({jeca:'JECA',limite:'LIMT',rio40:'R40G',cangaceiro:'CANG',aruanda:'ARUA',pagador:'PAGD'}[id]);

  React.useEffect(() => {
    if (!document.getElementById('ev-css')) {
      const s = document.createElement('style'); s.id = 'ev-css'; s.textContent = EV_CSS;
      document.head.appendChild(s);
    }
  }, []);

  // Current query: Q-042 with 9 candidates, 5 graded so far
  const initialGrades = [2, 3, 1, 3, 2, null, null, null, null]; // graded so far
  const [grades, setGrades] = React.useState(initialGrades);
  const [cur, setCur] = React.useState(5); // next ungraded
  const [blind, setBlind] = React.useState(false);
  const [showOther, setShowOther] = React.useState(true);
  const [filter, setFilter] = React.useState('todas');

  // Second annotator (Júlia Reis) mock grades for comparison
  const otherGrades = [2, 3, 1, 2, 2, 1, 1, 1, 0];

  // Keyboard: 0/1/2/3 grade & advance, S skip, j/k nav
  React.useEffect(() => {
    const onKey = (e) => {
      if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (['0','1','2','3'].includes(e.key)) {
        e.preventDefault();
        const g = parseInt(e.key, 10);
        setGrades(arr => { const next = [...arr]; next[cur] = g; return next; });
        setCur(c => Math.min(results.length - 1, c + 1));
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        setGrades(arr => { const next = [...arr]; next[cur] = 's'; return next; });
        setCur(c => Math.min(results.length - 1, c + 1));
      } else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); setCur(c => Math.min(results.length - 1, c + 1)); }
      else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); setCur(c => Math.max(0, c - 1)); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cur, results.length]);

  // Metrics
  const gradedCount = grades.filter(g => g !== null && g !== 's').length;
  const topNGrades = grades.slice(0, 5).filter(g => g !== null && g !== 's');
  const pAt5 = topNGrades.length ? topNGrades.filter(g => g >= 2).length / topNGrades.length : 0;
  const topNGrades3 = grades.slice(0, 3).filter(g => g !== null && g !== 's');
  const pAt3 = topNGrades3.length ? topNGrades3.filter(g => g >= 2).length / topNGrades3.length : 0;

  // NDCG@5 with grades as gains
  const dcg5 = grades.slice(0,5).reduce((sum, g, i) => {
    if (g === null || g === 's') return sum;
    return sum + (Math.pow(2, g) - 1) / Math.log2(i + 2);
  }, 0);
  const idealSorted = grades.slice(0,5).filter(g => g !== null && g !== 's').sort((a,b) => b - a);
  const idcg5 = idealSorted.reduce((sum, g, i) => sum + (Math.pow(2, g) - 1) / Math.log2(i + 2), 0);
  const ndcg5 = idcg5 > 0 ? dcg5 / idcg5 : 0;

  // grade histogram (current)
  const hist = {0:0, 1:0, 2:0, 3:0, s:0};
  grades.forEach(g => { if (g !== null) hist[g] = (hist[g] || 0) + 1; });
  const histMax = Math.max(1, ...Object.values(hist));

  // Agreement
  const compared = grades.map((g, i) => g !== null && g !== 's' ? {g, o: otherGrades[i]} : null).filter(Boolean);
  const sameCount = compared.filter(x => x.g === x.o).length;
  const closeCount = compared.filter(x => x.g !== x.o && Math.abs(x.g - x.o) <= 1).length;
  const farCount = compared.filter(x => x.g !== x.o && Math.abs(x.g - x.o) >= 2).length;
  const kappa = 0.71; // mock
  const agreePct = compared.length ? (sameCount / compared.length * 100).toFixed(0) : '—';

  // RENDER
  return (
    <div className="ev-app">
      {/* ADMIN STRIP */}
      <div className="ev-admin">
        <div className="l">
          <span className="badge">ADMIN</span>
          <span>Modo interno · julgamento de relevância</span>
        </div>
        <div className="r">
          <span>run · <b>eval-2026-04</b></span>
          <span>seed · <b>0xC1NE</b></span>
          <span>annotators · <b>2</b></span>
          <span>iaa · κ <b>{kappa.toFixed(2)}</b></span>
        </div>
      </div>

      {/* HEADER */}
      <div className="ev-top">
        <div className="l">
          <div className="brand">
            <FXMark size={20} />
            <span className="n">Mojica</span>
            <span className="sub">Eval set builder · v0.3</span>
          </div>
          <span className="div"></span>
          <div className="crumb">
            <span className="seg">Eval</span>
            <span className="sep">/</span>
            <span className="seg cur">run eval-2026-04</span>
            <span className="sep">/</span>
            <span className="runid">Q-042</span>
          </div>
        </div>
        <div className="session-counter">
          <span className="n">42</span>
          <span className="k">/ 200 julgadas</span>
          <span className="bar" style={{'--p': '21%'}}></span>
          <span className="k">21%</span>
        </div>
        <div className="r">
          <button className="fx-icbtn" title="Filter"><I.filter /></button>
          <button className="fx-icbtn" title="Sort"><I.sort /></button>
          <button className="fx-icbtn" title="Export"><I.download /></button>
          <button className="fx-icbtn" title="Settings"><I.settings /></button>
          <span className="div"></span>
          <div className="me">
            <div className="av">RG</div>
            <div>
              <div className="name">Rafael Gonzaga</div>
              <div className="role">curador · annotator-A</div>
            </div>
          </div>
        </div>
      </div>

      {/* BODY */}
      <div className="ev-body">
        {/* LEFT — QUEUE */}
        <aside className="ev-lp">
          <div className="head">
            <span>Fila de queries</span>
            <span className="v">42 / 200</span>
          </div>
          <div className="filter">
            <span className="ico"><I.search /></span>
            <input placeholder="Filtrar queries…" />
          </div>
          <div className="lpf-tabs">
            <span className={'t' + (filter==='pendentes' ? ' on' : '')} onClick={()=>setFilter('pendentes')}>
              Pendentes <span className="ct">11</span>
            </span>
            <span className={'t' + (filter==='todas' ? ' on' : '')} onClick={()=>setFilter('todas')}>
              Todas <span className="ct">200</span>
            </span>
            <span className={'t' + (filter==='conflito' ? ' on' : '')} onClick={()=>setFilter('conflito')}>
              Conflito <span className="ct">7</span>
            </span>
          </div>

          <div className="ev-q-list">
            {EV_QUEUES.map(q => (
              <div key={q.id} className={'ev-q' + (q.cur ? ' cur' : '')}>
                <span className={'stat' + (q.done === 9 ? ' done' : '')}>
                  {q.done === 9 && <I.check />}
                </span>
                <div className="body">
                  <div className="qrow">
                    <span className="id">{q.id}</span>
                    <span className="lang">{q.lang}</span>
                    <span className="src">{q.src}</span>
                    {q.cur && <span style={{marginLeft:'auto',color:FX.ac,fontFamily:'var(--mono)'}}>{gradedCount}/9</span>}
                  </div>
                  <div className="text">{q.text}</div>
                  <div className="progress">
                    <div className="pips">
                      {q.cur
                        ? grades.map((g, i) => <span key={i} className={'pp ' + (g === null ? 'gp' : g === 's' ? 'gp' : 'g'+g)}></span>)
                        : q.grades
                          ? q.grades.map((g, i) => <span key={i} className={'pp g'+g}></span>)
                          : Array.from({length: 9}, (_, i) => <span key={i} className="pp gp"></span>)
                      }
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Done section */}
            <div className="head" style={{paddingTop:18}}>
              <span>Concluídas · run atual</span>
              <span className="v">9 / 200</span>
            </div>
            {EV_DONE.map(q => (
              <div key={q.id} className="ev-q">
                <span className="stat done"><I.check /></span>
                <div className="body">
                  <div className="qrow">
                    <span className="id">{q.id}</span>
                    <span className="lang">pt</span>
                  </div>
                  <div className="text">{q.text}</div>
                  <div className="progress">
                    <div className="pips">
                      {q.grades.map((g, i) => <span key={i} className={'pp g'+g}></span>)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="controls">
            <div className={'toggle' + (blind ? ' on' : '')} onClick={() => setBlind(!blind)}>
              <span>Modo cego <span style={{fontFamily:'var(--mono)',color:FX.muted,marginLeft:4,fontSize:11}}>(score oculto)</span></span>
              <span className="swt"></span>
            </div>
            <div className={'toggle' + (showOther ? ' on' : '')} onClick={() => setShowOther(!showOther)}>
              <span>Comparar com JR</span>
              <span className="swt"></span>
            </div>
          </div>
        </aside>

        {/* CENTER — current query + candidates */}
        <section className="ev-cp">
          <div className="ev-q-card">
            <div className="ll">
              <span className="qid">Q-042</span>
              <span className="src">manual · rafael</span>
              <span>criada há 3 dias</span>
              <span>·</span>
              <span>pt-BR</span>
              <span className="ann">
                <span>annotators</span>
                <span style={{display:'inline-flex',gap:0,marginLeft:4}}>
                  <span style={{width:18,height:18,borderRadius:'50%',background:`linear-gradient(135deg,${FX.ac},${FX.pink})`,color:'#fff',display:'flex',alignItems:'center',justifyContent:'center',fontFamily:'var(--mono)',fontSize:9,fontWeight:700,marginRight:-4,border:`1.5px solid ${FX.panel}`}}>RG</span>
                  <span style={{width:18,height:18,borderRadius:'50%',background:'linear-gradient(135deg,#5CCB91,#2E8A5D)',color:'#fff',display:'flex',alignItems:'center',justifyContent:'center',fontFamily:'var(--mono)',fontSize:9,fontWeight:700,border:`1.5px solid ${FX.panel}`}}>JR</span>
                </span>
              </span>
            </div>
            <h1>
              <span className="kind">› query</span>
              duas pessoas conversando ao ar livre
            </h1>
            <div className="meta">
              <span className="pill"><span className="dot" style={{background:FX.ac}}></span>texto</span>
              <span>Top-k recuperados <b>k=9</b></span>
              <span>híbrido <b>sem 0.70 · bm25 0.30</b></span>
              <span>rerank <b>mxbai-rerank-l</b></span>
              <span>MMR <b>λ 0.5</b></span>
              <span>latência <b>231ms</b></span>
            </div>
          </div>

          <div className="legend">
            <span className="lab">grade:</span>
            <span className="chip g0">0 · irrelevante</span>
            <span className="chip g1">1 · vagamente</span>
            <span className="chip g2">2 · relevante</span>
            <span className="chip g3">3 · altamente</span>
            <span className="chip sk">S · não julgável</span>
            <span style={{marginLeft:'auto',color:FX.muted}}>
              {gradedCount} / 9 julgados · ungraded ↓
            </span>
          </div>

          <div className="ev-rows">
            {results.map((rr, i) => {
              const ff = byId[rr.film];
              const g = grades[i];
              const og = otherGrades[i];
              const disagrees = g !== null && g !== 's' && Math.abs(g - og) >= 2;
              return (
                <div key={rr.id}
                     className={'ev-row' + (i === cur ? ' cur' : '') + (g !== null ? ' graded' : '')}
                     onClick={() => setCur(i)}>
                  <div className="rank">
                    <span>{String(i+1).padStart(2,'0')}</span>
                    <span className="rk">rank</span>
                  </div>
                  <div className="grades">
                    {[0,1,2,3].map(n => (
                      <button key={n}
                              className={'gb' + (g === n ? ' on'+n : '') + (g !== null && g !== 's' && g !== n ? ' dim' : '')}
                              onClick={(e) => { e.stopPropagation(); setGrades(arr => { const x = [...arr]; x[i] = n; return x; }); }}>
                        {n}
                      </button>
                    ))}
                    <button className={'gb k' + (g === 's' ? ' ons' : '') + (g !== null && g !== 's' ? ' dim' : '')}
                            onClick={(e) => { e.stopPropagation(); setGrades(arr => { const x = [...arr]; x[i] = 's'; return x; }); }}>S</button>
                  </div>
                  <div className="thumb" style={{backgroundImage: `url(${rr.kf})`}}></div>
                  <div className="body">
                    <div className="head1">
                      <span className="id">{filmKey(rr.film)}-{String(rr.cena).padStart(3,'0')}</span>
                      <span className="filmpill">
                        <span className="dot" style={{background: FX_FILM[rr.film]}}></span>
                        <span className="nm">{ff.title}</span>
                        <span className="yr">{ff.year}</span>
                      </span>
                      <span className="tc">{rr.tc}</span>
                    </div>
                    <div className="desc">{rr.desc}</div>
                    <div className="tags">
                      {rr.tags.slice(0,5).map((t, j) => (
                        <span key={j} className="t" style={{color: (t==='duas-pessoas'||t==='exterior') ? FX.ac : FX.faint}}>
                          {t}{j < Math.min(rr.tags.length, 5) - 1 ? ' ·' : ''}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="right">
                    <span className={'score' + (blind ? ' blind' : '')}>{blind ? '••••' : rr.score.toFixed(3)}</span>
                    <span className={'scbar' + (blind ? ' blind' : '')} style={{'--p': blind ? '50%' : `${(rr.score*100).toFixed(0)}%`}}></span>
                    <span className="src">
                      {showOther && og !== undefined ? (
                        <span style={{color: disagrees ? FX.red : FX.muted}}>
                          JR · {og}
                        </span>
                      ) : 'sem⊕bm25⊕rk'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* RIGHT — METRICS */}
        <aside className="ev-rp">
          <div className="head">
            <h3>Métricas · Q-042</h3>
            <div className="acts">
              <button className="fx-icbtn sm"><I.more /></button>
            </div>
          </div>
          <div className="inner">
            <div className="ev-bigmets">
              <div className="ev-met">
                <span className="lab">Precision@5</span>
                <div className="val">
                  <span className="n">{(pAt5*100).toFixed(0)}<span style={{fontSize:13}}>%</span></span>
                  <span className="d">@ 5 grades ≥ 2</span>
                </div>
                <span className="bar" style={{'--p': `${(pAt5*100).toFixed(0)}%`}}></span>
              </div>
              <div className="ev-met fused">
                <span className="lab">nDCG@5</span>
                <div className="val">
                  <span className="n">{ndcg5.toFixed(2)}</span>
                  <span className="d">vs ideal</span>
                  <span className="delta up">+.07</span>
                </div>
                <span className="bar" style={{'--p': `${(ndcg5*100).toFixed(0)}%`}}></span>
              </div>
              <div className="ev-met">
                <span className="lab">Precision@3</span>
                <div className="val">
                  <span className="n">{(pAt3*100).toFixed(0)}<span style={{fontSize:13}}>%</span></span>
                  <span className="d">@ 3</span>
                </div>
                <span className="bar" style={{'--p': `${(pAt3*100).toFixed(0)}%`}}></span>
              </div>
              <div className="ev-met warn">
                <span className="lab">Inversões</span>
                <div className="val">
                  <span className="n">3</span>
                  <span className="d">de 10 pares</span>
                </div>
                <span className="bar" style={{'--p': '30%'}}></span>
              </div>
            </div>

            <div className="ev-sect">
              <span>Histograma de grades · Q-042</span>
              <span className="v">{gradedCount} / 9</span>
            </div>
            <div className="ev-hist">
              {[3,2,1,0].map(n => (
                <div key={n} className={'row g'+n}>
                  <span className={'g'+n}>g={n}</span>
                  <span className="track" style={{'--p': `${(hist[n]/histMax*100).toFixed(0)}%`}}></span>
                  <span className="ct">{hist[n]}</span>
                </div>
              ))}
              <div className="row gs">
                <span className="gs">skip</span>
                <span className="track" style={{'--p': `${(hist.s/histMax*100).toFixed(0)}%`}}></span>
                <span className="ct">{hist.s}</span>
              </div>
            </div>

            <div className="ev-sect">
              <span>Inter-annotator · vs JR</span>
              <a>detalhes</a>
            </div>
            <div className="ev-iaa">
              <div className="h">
                <div className="l">
                  <div className="av">JR</div>
                  <span>Júlia Reis</span>
                </div>
                <div className="r"><b>{agreePct}%</b> concordância</div>
              </div>
              <div className="kappa">
                <span className="label">Cohen's κ</span>
                <span className="val">{kappa.toFixed(2)}</span>
                <span className="qual">substancial</span>
              </div>
              <div className="matrix">
                <span className="h-cell"></span>
                <span className="h-cell">0</span>
                <span className="h-cell">1</span>
                <span className="h-cell">2</span>
                <span className="h-cell">3</span>
                <span className="h-cell">JR</span>

                <span className="h-cell">0</span>
                <span className="c diag hi">0</span>
                <span className="c">0</span>
                <span className="c">0</span>
                <span className="c">0</span>
                <span className="c">0</span>

                <span className="h-cell">1</span>
                <span className="c">0</span>
                <span className="c diag hi">1</span>
                <span className="c mid">0</span>
                <span className="c">0</span>
                <span className="c">1</span>

                <span className="h-cell">2</span>
                <span className="c">0</span>
                <span className="c mid">0</span>
                <span className="c diag hi">2</span>
                <span className="c">0</span>
                <span className="c">2</span>

                <span className="h-cell">3</span>
                <span className="c">0</span>
                <span className="c">0</span>
                <span className="c mid">1</span>
                <span className="c diag hi">1</span>
                <span className="c">2</span>

                <span className="h-cell" style={{fontWeight:600,color:FX.t}}>RG</span>
                <span className="c">0</span>
                <span className="c">1</span>
                <span className="c">3</span>
                <span className="c">1</span>
                <span className="c diag" style={{background:FX.bd,color:FX.t}}>5</span>
              </div>
            </div>

            <div className="ev-sect">
              <span>Sessão · run eval-2026-04</span>
              <span className="v">42:12</span>
            </div>
            <div className="ev-sess">
              <div className="row"><span className="k">Queries julgadas</span><span className="v"><b style={{color:FX.t}}>42</b> / 200</span></div>
              <div className="row"><span className="k">Tempo médio · query</span><span className="v">2m 18s</span></div>
              <div className="row"><span className="k">Mais rápida</span><span className="v">0m 47s</span></div>
              <div className="row"><span className="k">Mais lenta</span><span className="v">5m 32s</span></div>
              <div className="row"><span className="k">Tempo nesta sessão</span><span className="v acc">42m 12s</span></div>
              <div className="row"><span className="k">Conflitos abertos</span><span className="v warn">7</span></div>
              <div className="row"><span className="k">P@5 médio · run</span><span className="v acc">0.68</span></div>
              <div className="row"><span className="k">nDCG@5 médio · run</span><span className="v acc">0.79</span></div>
              <div className="row"><span className="k">ETA · run</span><span className="v">~ 7h restante</span></div>
            </div>

            <div className="ev-saverow">
              <button className="ev-save">
                <span>Salvar e avançar · Q-043</span>
                <span className="kbd">⌘ ⏎</span>
              </button>
              <button className="ev-skip">
                <span>Pular query (irrecuperável)</span>
                <span style={{fontFamily:'var(--mono)',fontSize:10,color:FX.muted}}>⇧S</span>
              </button>
            </div>
          </div>
        </aside>
      </div>

      {/* BOTTOM STATUS */}
      <div className="ev-bot">
        <div style={{display:'flex',alignItems:'center'}}>
          <span className="mode">eval</span>
          <div className="keys">
            <span className="k"><b>0/1/2/3</b> grade</span>
            <span className="k"><b>S</b> skip</span>
            <span className="k"><b>j/k</b> nav</span>
            <span className="k"><b>⌘⏎</b> salvar e avançar</span>
            <span className="k"><b>⌘E</b> exportar TREC</span>
            <span className="k"><b>?</b> ajuda</span>
          </div>
        </div>
        <div className="r">
          <span>row <b>{String(cur+1).padStart(2,'0')}/09</b></span>
          <span>Q <b>42/200</b></span>
          <span>session <b>42:12</b></span>
          <span>idx <b style={{color: FX.green}}>ok</b></span>
          <span>v0.3.0</span>
        </div>
      </div>
    </div>
  );
}

window.ScreenEval = ScreenEval;
window.EV_CSS = EV_CSS;
