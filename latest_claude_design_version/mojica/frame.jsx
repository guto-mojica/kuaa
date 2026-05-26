// Cinemateca Mojica — Frame.io branch
// 3-pane interconnected mechanic preserved. Visual language reskinned as
// a creative-pro media review tool: dark blue panels, cyan accent,
// rounded corners, comment-thread inspector, scrubber-timeline at the
// bottom of the center pane.

const F_PALETTE = {
  bg:        '#0F1216',
  panel:     '#181C22',
  raised:    '#1F242C',
  hover:     '#232831',
  border:    '#262B33',
  border2:   '#363C46',
  text:      '#F1F3F5',
  text2:     '#B1B5BD',
  muted:     '#6E7681',
  faint:     '#454A54',
  accent:    '#4DAFFC',
  accentDim: '#2A6FB3',
  accentBg:  'rgba(77,175,252,0.12)',
  coral:     '#FF6B4A',
  green:     '#4CC38A',
  yellow:    '#F2C94C',
};

const F_CSS = `
.f-app, .f-app * { box-sizing: border-box; }
.f-app *::-webkit-scrollbar { width: 8px; height: 8px; }
.f-app *::-webkit-scrollbar-thumb { background: ${F_PALETTE.border2}; border-radius: 4px; }
.f-app *::-webkit-scrollbar-track { background: transparent; }

.f-app {
  --bg: ${F_PALETTE.bg};
  --panel: ${F_PALETTE.panel};
  --raised: ${F_PALETTE.raised};
  --hover: ${F_PALETTE.hover};
  --bd: ${F_PALETTE.border};
  --bd2: ${F_PALETTE.border2};
  --t: ${F_PALETTE.text};
  --t2: ${F_PALETTE.text2};
  --muted: ${F_PALETTE.muted};
  --faint: ${F_PALETTE.faint};
  --ac: ${F_PALETTE.accent};
  --ac-dim: ${F_PALETTE.accentDim};
  --ac-bg: ${F_PALETTE.accentBg};
  --coral: ${F_PALETTE.coral};
  --green: ${F_PALETTE.green};
  --yellow: ${F_PALETTE.yellow};
  --sans: 'Geist', 'Söhne', system-ui, sans-serif;
  --mono: 'Geist Mono', 'JetBrains Mono', monospace;
  display: grid;
  grid-template-rows: 56px 1fr;
  height: 100vh; width: 100vw;
  background: var(--bg); color: var(--t);
  font-family: var(--sans); font-size: 13px; line-height: 1.5;
  letter-spacing: -0.003em;
  font-feature-settings: 'ss01' on, 'ss02' on;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

/* TOPBAR */
.f-top {
  display:flex; align-items:center; justify-content: space-between;
  padding: 0 18px; border-bottom: 1px solid var(--bd);
  background: var(--panel);
}
.f-top .left { display:flex; align-items:center; gap: 14px; }
.f-top .brand { display:flex; align-items:center; gap: 9px; }
.f-top .brand-name {
  font-weight: 600; font-size: 14px; color: var(--t); letter-spacing: -0.01em;
}
.f-top .divv { width: 1px; height: 22px; background: var(--bd); margin: 0 4px; }
.f-top .crumb {
  display:flex; align-items:center; gap: 8px;
  font-size: 13px; color: var(--t2);
}
.f-top .crumb .seg.cur { color: var(--t); font-weight: 500; }
.f-top .crumb .sep { color: var(--faint); font-size: 12px; }
.f-top .crumb .pill {
  font-family: var(--mono); font-size: 10.5px;
  padding: 2px 7px; background: var(--raised); border-radius: 3px;
  color: var(--t2); letter-spacing: 0.02em;
}
.f-top .center {
  display:flex; align-items:center; gap: 4px;
  background: var(--raised); border-radius: 6px; padding: 4px;
}
.f-top .center .seg-tab {
  padding: 5px 12px; border-radius: 4px;
  font-size: 12.5px; color: var(--t2); cursor: pointer; font-weight: 500;
}
.f-top .center .seg-tab.on { background: var(--hover); color: var(--t); }
.f-top .center .seg-tab .pip {
  display:inline-block; background: var(--coral); color: #1a0d09;
  font-size: 10px; padding: 0 5px; border-radius: 7px;
  margin-left: 5px; font-weight: 600; font-family: var(--mono);
}
.f-top .right { display:flex; align-items:center; gap: 10px; }
.f-top .iconbtn {
  width: 32px; height: 32px; border-radius: 5px;
  display:flex; align-items:center; justify-content:center;
  background: transparent; border: none; color: var(--t2);
  cursor: pointer; font-family: var(--mono); font-size: 14px;
}
.f-top .iconbtn:hover { background: var(--hover); color: var(--t); }
.f-top .iconbtn.has-badge { position: relative; }
.f-top .iconbtn .nb {
  position:absolute; top: 5px; right: 5px; width: 6px; height: 6px;
  background: var(--coral); border-radius: 50%;
}
.f-top .share {
  display:flex; align-items:center; gap: 7px;
  padding: 6px 14px; border-radius: 5px;
  background: var(--ac); color: #00131F; font-weight: 600;
  font-size: 12.5px; cursor: pointer; border: none;
}
.f-top .share:hover { background: #6DBEFD; }
.f-top .ver {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  padding: 5px 9px; border: 1px solid var(--bd2); border-radius: 5px;
  display:flex; align-items:center; gap: 6px;
}
.f-top .ver .dot {
  width: 6px; height: 6px; background: var(--green); border-radius: 50%;
}
.f-top .avatar {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, var(--ac), var(--coral));
  display:flex; align-items:center; justify-content:center;
  font-size: 11px; font-weight: 600; color: #0F1216; letter-spacing: 0;
}

/* BODY */
.f-body {
  display: grid;
  grid-template-columns: 252px 1fr 372px;
  overflow: hidden; min-height: 0;
}

/* LEFT */
.f-lp {
  border-right: 1px solid var(--bd);
  display:flex; flex-direction:column; background: var(--panel); overflow: hidden;
}
.f-lp .sect {
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 16px 8px;
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}
.f-lp .sect .add {
  width: 18px; height: 18px; border-radius: 4px;
  background: transparent; border: 1px solid var(--bd2); color: var(--t2);
  display:flex; align-items:center; justify-content:center;
  cursor:pointer; font-size: 14px; line-height: 1;
}
.f-lp .sect .add:hover { border-color: var(--ac); color: var(--ac); }
.f-lp .filter {
  margin: 0 12px 6px; padding: 7px 10px; border-radius: 6px;
  background: var(--bg); border: 1px solid var(--bd);
  display:flex; align-items:center; gap: 8px;
  font-size: 12px; color: var(--muted);
}
.f-lp .filter input {
  flex: 1; background: transparent; border: none; outline: none;
  font: inherit; color: var(--t);
}
.f-lp .filter .kbd {
  font-family: var(--mono); font-size: 10px; padding: 1px 5px;
  border: 1px solid var(--bd2); border-radius: 3px; color: var(--muted);
}

.f-lp .tree { padding: 2px 8px 8px; overflow-y: auto; }
.f-prj {
  display:grid; grid-template-columns: 22px 16px 1fr auto;
  align-items:center; gap: 6px;
  padding: 6px 8px; border-radius: 5px;
  cursor: pointer; position: relative;
}
.f-prj:hover { background: var(--hover); }
.f-prj.active { background: var(--ac-bg); }
.f-prj.active .name { color: var(--ac); font-weight: 500; }
.f-prj .arr { color: var(--muted); font-size: 10px; text-align: center; }
.f-prj .ico { color: var(--muted); font-size: 14px; display:flex; align-items:center; }
.f-prj.active .ico { color: var(--ac); }
.f-prj .name { font-size: 13px; color: var(--t2); }
.f-prj.has-sel .name { color: var(--t); }
.f-prj .ct {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.f-prj .meta {
  grid-column: 3 / 5;
  display:flex; align-items:center; gap: 8px;
  font-size: 11px; color: var(--muted); margin-top: 2px;
}
.f-prj .progress {
  flex: 1; height: 4px; background: var(--bd); border-radius: 2px;
  position: relative; overflow: hidden;
}
.f-prj .progress::before {
  content:''; position:absolute; left:0; top:0; bottom:0;
  width: var(--p, 0%); background: var(--ac); border-radius: 2px;
}
.f-prj.has-sel .progress::before { background: var(--coral); }
.f-prj.proc .name { color: var(--yellow); }
.f-prj.proc .arr { color: var(--yellow); }
.f-prj.proc .progress::before { background: var(--yellow); }
.f-prj .selptr {
  grid-column: 2 / 5;
  display: none; align-items: center; gap: 6px; margin-top: 6px;
  padding: 5px 7px; background: var(--raised); border-radius: 4px;
  font-family: var(--mono); font-size: 10px; color: var(--ac);
}
.f-prj.has-sel .selptr { display: flex; }
.f-prj .selptr .thumb {
  width: 24px; height: 18px; border-radius: 2px;
  background-size: cover; background-position: center;
}

/* shared views section */
.f-views { padding: 0 8px 8px; }
.f-view {
  display:grid; grid-template-columns: 16px 1fr auto; gap: 8px;
  align-items: center; padding: 6px 8px; border-radius: 5px;
  cursor: pointer; font-size: 13px; color: var(--t2);
}
.f-view:hover { background: var(--hover); }
.f-view .ico { color: var(--muted); font-size: 13px; }
.f-view .ct {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}

.f-lp .lp-foot {
  margin-top: auto; border-top: 1px solid var(--bd);
  padding: 10px 14px; display:flex; align-items:center; justify-content:space-between;
  font-size: 11.5px;
}
.f-lp .lp-foot .left { display:flex; align-items:center; gap: 8px; }
.f-lp .lp-foot .dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--green);
}
.f-lp .lp-foot .right { color: var(--muted); font-family: var(--mono); font-size: 10.5px; }

/* CENTER */
.f-cp { display:flex; flex-direction:column; min-width: 0; overflow: hidden; background: var(--bg); }

.f-search {
  padding: 16px 24px 12px; border-bottom: 1px solid var(--bd);
}
.f-srow {
  display:flex; align-items:center; gap: 10px;
  background: var(--panel); border: 1px solid var(--bd);
  padding: 8px 10px 8px 14px; border-radius: 7px;
  transition: border-color .12s, background .12s;
}
.f-srow:focus-within { border-color: var(--ac); background: var(--raised); box-shadow: 0 0 0 3px var(--ac-bg); }
.f-srow .ico { color: var(--muted); font-size: 15px; }
.f-srow input {
  flex:1; background: transparent; border:none; outline:none;
  font: inherit; font-size: 14px; color: var(--t);
}
.f-srow input::placeholder { color: var(--muted); }
.f-srow .kbd {
  font-family: var(--mono); font-size: 10px; padding: 2px 6px;
  border: 1px solid var(--bd2); border-radius: 4px; color: var(--muted);
}
.f-srow .submit {
  padding: 7px 13px; border-radius: 5px;
  background: var(--ac); color: #00131F; font-weight: 600;
  font-size: 12px; cursor: pointer; border: none;
  display:flex; align-items:center; gap: 6px;
}

.f-modes {
  display:flex; align-items:center; gap: 10px; margin-top: 12px;
  flex-wrap: wrap;
}
.f-chip {
  display:flex; align-items:center; gap: 6px;
  padding: 6px 11px; border-radius: 5px;
  background: var(--panel); border: 1px solid var(--bd);
  font-size: 12px; color: var(--t2); cursor: pointer;
  font-weight: 500;
}
.f-chip:hover { border-color: var(--bd2); color: var(--t); }
.f-chip.on { background: var(--ac-bg); border-color: var(--ac-dim); color: var(--ac); }
.f-chip .ico { font-size: 13px; }
.f-modes .div { width: 1px; height: 20px; background: var(--bd); margin: 0 4px; }
.f-knob {
  display:flex; align-items:center; gap: 6px; font-size: 11.5px; color: var(--muted);
}
.f-knob .k { color: var(--faint); font-family: var(--mono); font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.06em; }
.f-knob .v {
  color: var(--t2); font-family: var(--mono); font-size: 11px; font-variant-numeric: tabular-nums;
  padding: 2px 7px; background: var(--panel); border-radius: 4px; border: 1px solid var(--bd);
}
.f-knob .v.acc { color: var(--ac); border-color: var(--ac-dim); }

.f-caption {
  display:flex; align-items:center; justify-content:space-between;
  padding: 12px 24px; border-bottom: 1px solid var(--bd);
  background: var(--bg);
}
.f-caption .left {
  display:flex; align-items:center; gap: 16px;
  font-size: 13px; color: var(--t);
}
.f-caption .left b { color: var(--ac); font-weight: 600; }
.f-caption .left .meta {
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  display:flex; align-items:center; gap: 10px;
}
.f-caption .left .meta b { color: var(--t); font-weight: 500; }
.f-caption .right { display:flex; align-items:center; gap: 8px; }
.f-caption .right .seg-tab {
  font-size: 11.5px; color: var(--muted); padding: 4px 8px; border-radius: 4px;
  cursor: pointer; font-weight: 500;
}
.f-caption .right .seg-tab.on { background: var(--hover); color: var(--t); }

/* RESULTS GRID */
.f-grid {
  flex: 1; overflow-y: auto;
  padding: 20px 24px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 18px;
  align-content: start;
}
.f-card {
  background: var(--panel); border: 1px solid var(--bd);
  border-radius: 8px; overflow: hidden;
  cursor: pointer; display:flex; flex-direction: column;
  transition: border-color .12s, transform .12s, box-shadow .12s;
}
.f-card:hover { border-color: var(--bd2); transform: translateY(-1px); }
.f-card.sel { border-color: var(--ac); box-shadow: 0 0 0 2px var(--ac-bg); }
.f-card .kf {
  width: 100%; aspect-ratio: 4/3;
  background: var(--bg) center/cover no-repeat;
  position: relative; filter: contrast(1.04) brightness(0.96);
}
.f-card .kf .tag-tl {
  position: absolute; top: 8px; left: 8px;
  display:flex; align-items:center; gap: 5px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(15,18,22,0.86); border-radius: 4px;
  backdrop-filter: blur(4px);
}
.f-card .kf .tag-tl .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
.f-card .kf .tag-bl {
  position: absolute; bottom: 8px; left: 8px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(15,18,22,0.86); border-radius: 4px;
  backdrop-filter: blur(4px); letter-spacing: 0.02em;
}
.f-card .kf .tag-tr {
  position: absolute; top: 8px; right: 8px;
  font-family: var(--mono); font-size: 11px; color: var(--ac);
  padding: 3px 7px; background: rgba(15,18,22,0.86); border-radius: 4px;
  backdrop-filter: blur(4px); font-weight: 600;
}
.f-card .kf .ann {
  position: absolute; bottom: 8px; right: 8px;
  display:flex; align-items:center; gap: 4px;
  font-family: var(--mono); font-size: 10px; color: var(--coral);
  padding: 3px 7px; background: rgba(15,18,22,0.86); border-radius: 4px;
}
.f-card .kf .ann .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--coral); }
.f-card .body { padding: 11px 12px 12px; display:flex; flex-direction: column; gap: 6px; }
.f-card .title {
  font-size: 13.5px; font-weight: 600; color: var(--t); letter-spacing: -0.005em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.f-card .sub {
  display:flex; align-items:center; gap: 8px;
  font-size: 11.5px; color: var(--muted);
}
.f-card .sub .film-pill {
  font-family: var(--mono); font-size: 10px;
  background: var(--raised); padding: 1px 6px; border-radius: 3px;
  color: var(--t2); font-weight: 500;
}
.f-card .sub .yr { font-family: var(--mono); font-size: 10.5px; }
.f-card .desc {
  font-size: 12px; line-height: 1.5; color: var(--t2);
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}
.f-card .footrow {
  display:flex; align-items:center; justify-content:space-between;
  padding-top: 4px; border-top: 1px solid var(--bd);
}
.f-card .footrow .tags { display:flex; gap: 4px; flex-wrap: wrap; }
.f-card .footrow .tag {
  font-family: var(--mono); font-size: 9.5px;
  padding: 1px 5px; border-radius: 3px;
  background: var(--raised); color: var(--t2);
}
.f-card .footrow .tag.m { color: var(--ac); background: var(--ac-bg); }
.f-card .footrow .acts { display:flex; gap: 4px; color: var(--muted); }
.f-card .footrow .acts .a { width: 20px; height: 20px; display:flex; align-items:center; justify-content:center; font-size: 12px; }

/* TIMELINE STRIP */
.f-timeline {
  border-top: 1px solid var(--bd); background: var(--panel);
  padding: 12px 24px 14px;
  display:flex; flex-direction:column; gap: 10px;
}
.f-timeline .head {
  display:flex; align-items:center; justify-content:space-between;
}
.f-timeline .ttitle {
  display:flex; align-items:center; gap: 10px;
  font-size: 12px; color: var(--t); font-weight: 500;
}
.f-timeline .ttitle .pill {
  font-family: var(--mono); font-size: 10px; padding: 2px 7px;
  background: var(--raised); border-radius: 3px; color: var(--t2);
}
.f-timeline .controls {
  display:flex; align-items:center; gap: 12px;
  font-family: var(--mono); font-size: 11px; color: var(--muted);
}
.f-timeline .controls .tc { color: var(--ac); font-variant-numeric: tabular-nums; }
.f-timeline .controls .btn {
  width: 26px; height: 24px; display:flex; align-items:center; justify-content:center;
  border-radius: 4px; background: var(--raised); color: var(--t2);
  cursor: pointer; font-size: 11px;
}
.f-timeline .controls .btn:hover { background: var(--hover); color: var(--t); }
.f-timeline .scrubrow {
  display: grid; grid-template-columns: 1fr; gap: 6px;
}
.f-timeline .scrub {
  position: relative; height: 56px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 5px;
  display:flex; gap: 0; overflow: hidden;
}
.f-timeline .seg {
  flex: 1; min-width: 0; height: 100%;
  background-size: cover; background-position: center;
  border-right: 1px solid rgba(0,0,0,0.5);
  filter: brightness(0.6) contrast(1.05);
  position: relative; cursor: pointer;
  transition: filter .15s;
}
.f-timeline .seg:hover { filter: brightness(0.9) contrast(1.05); }
.f-timeline .seg.match::after {
  content:''; position:absolute; left:0; right:0; bottom: 0;
  height: 3px; background: var(--ac);
}
.f-timeline .seg.sel { filter: brightness(1.0) contrast(1.1); }
.f-timeline .seg.sel::before {
  content:''; position: absolute; inset: 0;
  outline: 2px solid var(--coral); outline-offset: -2px;
  z-index: 2;
}
.f-timeline .seg.sel::after {
  background: var(--coral); height: 4px;
}
.f-timeline .ticks {
  display:grid; grid-template-columns: repeat(8, 1fr);
  font-family: var(--mono); font-size: 9.5px; color: var(--faint);
  letter-spacing: 0.02em;
}
.f-timeline .ticks span { text-align: left; }

/* RIGHT PANE — Activity / Inspector */
.f-rp {
  border-left: 1px solid var(--bd); background: var(--panel);
  display:flex; flex-direction:column; overflow: hidden;
}
.f-rp .htabs {
  display:flex; align-items:center; padding: 0 12px;
  border-bottom: 1px solid var(--bd);
  gap: 4px;
}
.f-rp .htabs .tab {
  padding: 13px 12px; font-size: 12.5px; color: var(--muted);
  cursor: pointer; position: relative; font-weight: 500;
}
.f-rp .htabs .tab.on { color: var(--t); }
.f-rp .htabs .tab.on::after {
  content:''; position:absolute; left: 8px; right: 8px; bottom: -1px;
  height: 2px; background: var(--ac);
}
.f-rp .htabs .tab .pip {
  font-family: var(--mono); font-size: 10px; padding: 0 5px;
  background: var(--raised); border-radius: 8px; margin-left: 4px;
  color: var(--t2);
}
.f-rp .htabs .gap { flex: 1; }
.f-rp .htabs .iconbtn {
  width: 28px; height: 28px; display:flex; align-items:center; justify-content:center;
  border-radius: 5px; color: var(--muted); cursor: pointer; font-size: 14px;
}
.f-rp .htabs .iconbtn:hover { background: var(--hover); color: var(--t); }

.f-rp .inner { padding: 14px 16px 18px; overflow-y: auto; flex: 1; }
.f-insp-kf {
  width: 100%; aspect-ratio: 16/10;
  background: var(--bg) center/cover no-repeat;
  border-radius: 6px; position: relative;
  filter: contrast(1.05) brightness(0.97);
  border: 1px solid var(--bd);
}
.f-insp-kf .pin {
  position: absolute; top: 14%; left: 22%;
  width: 22px; height: 22px; border-radius: 50%;
  background: var(--coral); color: #fff;
  display:flex; align-items:center; justify-content:center;
  font-size: 11px; font-weight: 700; box-shadow: 0 0 0 3px rgba(255,107,74,0.25);
  cursor: pointer;
}
.f-insp-kf .pc {
  position: absolute; bottom: 10px; left: 10px;
  display:flex; align-items:center; gap: 5px;
  font-family: var(--mono); font-size: 10px; color: #fff;
  padding: 3px 7px; background: rgba(15,18,22,0.86); border-radius: 4px;
}
.f-insp-kf .pc .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--coral); }

.f-insp-meta { display:flex; align-items:flex-start; justify-content:space-between; gap: 8px; margin-top: 12px; }
.f-insp-meta .left h3 {
  margin: 0; font-size: 16.5px; font-weight: 600;
  color: var(--t); letter-spacing: -0.012em; line-height: 1.2;
}
.f-insp-meta .left .at {
  display:flex; align-items:center; gap: 8px;
  margin-top: 5px; font-size: 11.5px; color: var(--muted);
}
.f-insp-meta .left .at .film-pill {
  font-family: var(--mono); font-size: 10.5px;
  background: var(--raised); padding: 1px 6px; border-radius: 3px; color: var(--t2);
}
.f-insp-meta .left .at .tc { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.f-insp-meta .status {
  display:flex; align-items:center; gap: 6px;
  font-family: var(--mono); font-size: 10.5px; padding: 4px 8px;
  background: rgba(76,195,138,0.10); color: var(--green);
  border-radius: 14px; font-weight: 500;
}
.f-insp-meta .status .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

/* comment thread */
.f-thread {
  margin-top: 18px; display:flex; flex-direction: column; gap: 12px;
}
.f-comment {
  display:grid; grid-template-columns: 30px 1fr; gap: 10px;
}
.f-comment .av {
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--raised); display:flex; align-items:center; justify-content:center;
  font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--t);
  letter-spacing: 0;
}
.f-comment.ai .av { background: linear-gradient(135deg, var(--ac), var(--ac-dim)); color: #00131F; }
.f-comment.curator .av { background: linear-gradient(135deg, var(--coral), #B0432B); color: #fff; }
.f-comment .bx { display:flex; flex-direction: column; gap: 4px; }
.f-comment .who {
  display:flex; align-items:baseline; gap: 7px;
  font-size: 12px;
}
.f-comment .who .name { color: var(--t); font-weight: 500; }
.f-comment .who .badge {
  font-family: var(--mono); font-size: 9.5px; padding: 1px 5px;
  border-radius: 3px; background: var(--raised); color: var(--t2);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.f-comment.ai .who .badge { background: var(--ac-bg); color: var(--ac); }
.f-comment .who .when {
  font-family: var(--mono); font-size: 10.5px; color: var(--muted);
}
.f-comment .body {
  font-size: 12.5px; line-height: 1.55; color: var(--t2);
  text-wrap: pretty;
}
.f-comment .replyrow {
  display:flex; align-items:center; gap: 11px; margin-top: 2px;
  font-size: 11px; color: var(--muted);
}
.f-comment .replyrow a { color: var(--muted); cursor: pointer; }
.f-comment .replyrow a:hover { color: var(--ac); }
.f-comment.pinned .who .badge {
  background: rgba(255,107,74,0.14); color: var(--coral);
}

/* retrieval bars */
.f-signals-card {
  margin-top: 18px; padding: 14px;
  background: var(--bg); border: 1px solid var(--bd);
  border-radius: 7px;
}
.f-signals-card .h {
  display:flex; align-items:baseline; justify-content:space-between;
  font-size: 11px; color: var(--muted); margin-bottom: 12px;
  font-weight: 500;
}
.f-signals-card .h .v { font-family: var(--mono); color: var(--t); font-variant-numeric: tabular-nums; }
.f-signals-card .row {
  display:grid; grid-template-columns: 80px 1fr 46px;
  align-items: center; gap: 10px; margin-bottom: 8px;
  font-family: var(--mono); font-size: 11px; color: var(--t2);
  font-variant-numeric: tabular-nums;
}
.f-signals-card .row .lab { color: var(--muted); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }
.f-signals-card .row .track { height: 5px; background: var(--bd); border-radius: 3px; position: relative; }
.f-signals-card .row .track::before {
  content:''; position:absolute; left: 0; top: 0; bottom: 0;
  width: var(--p); background: var(--ac-dim); border-radius: 3px;
}
.f-signals-card .row .v { text-align: right; color: var(--t); }
.f-signals-card .row.fused .track::before { background: var(--ac); }
.f-signals-card .row.fused .lab { color: var(--ac); }

/* rimas mini */
.f-rimas-card {
  margin-top: 14px; padding: 12px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 7px;
}
.f-rimas-card .h {
  display:flex; align-items:baseline; justify-content:space-between;
  font-size: 11px; color: var(--muted); margin-bottom: 10px; font-weight: 500;
}
.f-rimas-card .h a { color: var(--ac); cursor: pointer; font-family: var(--mono); font-size: 10.5px; }
.f-rimas-card .grid3 {
  display:grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
}
.f-rimas-card .ry { display:flex; flex-direction: column; gap: 4px; cursor: pointer; }
.f-rimas-card .ry .kf {
  width: 100%; aspect-ratio: 4/3; border-radius: 4px;
  background-size: cover; background-position: center;
  filter: contrast(1.05) brightness(0.95);
  border: 1px solid var(--bd); transition: border-color .12s;
}
.f-rimas-card .ry:hover .kf { border-color: var(--ac); }
.f-rimas-card .ry .lab {
  font-size: 10.5px; color: var(--t2); line-height: 1.2;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.f-rimas-card .ry .lab b { color: var(--ac); font-family: var(--mono); font-weight: 500; }

/* tags row */
.f-tagrow { display:flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; }
.f-tagrow .t {
  font-family: var(--mono); font-size: 10.5px; padding: 3px 8px;
  background: var(--raised); border-radius: 4px; color: var(--t2);
  cursor: pointer;
}
.f-tagrow .t.m { color: var(--ac); background: var(--ac-bg); }
.f-tagrow .add {
  font-family: var(--mono); font-size: 10.5px; padding: 3px 8px;
  background: transparent; border: 1px dashed var(--bd2); color: var(--muted);
  border-radius: 4px; cursor: pointer;
}

/* comment input */
.f-comment-input {
  margin-top: 16px; padding: 10px 12px;
  background: var(--bg); border: 1px solid var(--bd); border-radius: 7px;
}
.f-comment-input textarea {
  width: 100%; min-height: 44px; resize: none;
  background: transparent; border: none; outline: none;
  font: inherit; font-size: 12.5px; color: var(--t);
}
.f-comment-input textarea::placeholder { color: var(--muted); }
.f-comment-input .row {
  display:flex; align-items:center; justify-content:space-between; margin-top: 6px;
}
.f-comment-input .row .tools {
  display:flex; align-items:center; gap: 4px; color: var(--muted); font-size: 14px;
}
.f-comment-input .row .tools .tool {
  width: 24px; height: 24px; display:flex; align-items:center; justify-content:center;
  border-radius: 4px; cursor: pointer;
}
.f-comment-input .row .tools .tool:hover { background: var(--hover); color: var(--t); }
.f-comment-input .row .post {
  padding: 5px 12px; border-radius: 5px;
  background: var(--ac); color: #00131F; font-weight: 600;
  font-size: 11px; border: none; cursor: pointer;
}
`;

function MojicaFrame() {
  const films = window.FILMS;
  const results = window.RESULTS;
  const byId = Object.fromEntries(films.map(f => [f.id, f]));
  const [sel, setSel] = React.useState(0);

  React.useEffect(() => {
    if (!document.getElementById('f-css')) {
      const s = document.createElement('style');
      s.id = 'f-css'; s.textContent = F_CSS;
      document.head.appendChild(s);
    }
  }, []);

  const r = results[sel];
  const f = byId[r.film];

  const matches = {};
  results.forEach(rr => matches[rr.film] = (matches[rr.film] || 0) + 1);
  const maxM = Math.max(...Object.values(matches), 1);

  const sigSem  = Math.min(0.96, r.score + 0.06);
  const sigBm25 = Math.max(0.04, r.score - 0.52);
  const sigRk   = Math.min(0.96, r.score - 0.01);
  const sigFu   = r.score;

  const rhymes = results.filter((x, i) => x.film !== r.film && i !== sel).slice(0, 3)
    .map((x, i) => ({...x, sim: (0.94 - i*0.04).toFixed(2)}));

  // For the timeline strip: show 24 segments mocked from the selected film
  // (we have ~412 scenes in jeca; just pick a sample). Highlight matched and selected.
  const timelineSegs = React.useMemo(() => {
    const allKfs = [
      'keyframes/kf-01-title.jpg', 'keyframes/kf-02-fence.jpg', 'keyframes/kf-03-horse.jpg',
      'keyframes/kf-04-cow.jpg',   'keyframes/kf-05-man-cow.jpg', 'keyframes/kf-06-women-hut.jpg',
      'keyframes/kf-07-woman-pot.jpg', 'keyframes/kf-08-woman-dark.jpg', 'keyframes/kf-09-bed.jpg',
      'keyframes/kf-10-shirt.jpg', 'keyframes/kf-11-mustache.jpg', 'keyframes/kf-12-mustache2.jpg',
      'keyframes/kf-13-conversation.jpg', 'keyframes/kf-14-brinquinho.jpg',
      'keyframes/kf-15-night-fence.jpg', 'keyframes/kf-16-flames.jpg',
      'keyframes/kf-17-smoke.jpg', 'keyframes/kf-18-night-fire.jpg',
    ];
    return Array.from({length: 26}, (_, i) => allKfs[i % allKfs.length]);
  }, []);
  const matchedIdx = [3, 7, 11, 16, 21];

  return (
    <div className="f-app">
      {/* TOPBAR */}
      <div className="f-top">
        <div className="left">
          <div className="brand">
            <svg width="20" height="20" viewBox="0 0 22 22" fill="none">
              <rect x="0.5" y="2.5" width="21" height="17" stroke={F_PALETTE.text} strokeWidth="1.2" rx="2"/>
              <rect x="2.5" y="4.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="2.5" y="9.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="2.5" y="14.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="17" y="4.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="17" y="9.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="17" y="14.5" width="2.5" height="2.5" rx="0.5" fill={F_PALETTE.text}/>
              <rect x="7" y="6" width="8" height="10" rx="1" fill={F_PALETTE.accent}/>
            </svg>
            <span className="brand-name">Mojica</span>
          </div>
          <span className="divv"></span>
          <div className="crumb">
            <span className="seg">Acervo</span>
            <span className="sep">/</span>
            <span className="seg cur">Buscar semântico</span>
            <span className="pill">v1.0</span>
          </div>
        </div>
        <div className="center">
          <span className="seg-tab">Cenas</span>
          <span className="seg-tab on">Buscar</span>
          <span className="seg-tab">Anotar</span>
          <span className="seg-tab">Rimas</span>
          <span className="seg-tab">Processamento<span className="pip">1</span></span>
        </div>
        <div className="right">
          <button className="iconbtn" title="Filter">⊟</button>
          <button className="iconbtn has-badge" title="Notifications">⌒<span className="nb"></span></button>
          <button className="iconbtn" title="Comments">⌶</button>
          <span className="ver"><span className="dot"></span>v1.0.0</span>
          <button className="share">Compartilhar acervo ⌃</button>
          <div className="avatar" title="Curator">RG</div>
        </div>
      </div>

      <div className="f-body">
        {/* LEFT */}
        <aside className="f-lp">
          <div className="sect"><span>Projetos · Acervo</span><button className="add">+</button></div>
          <div className="filter">
            <span>⌕</span>
            <input placeholder="Filtrar filmes…" />
            <span className="kbd">/</span>
          </div>
          <div className="tree">
            {films.map(film => {
              const cnt = matches[film.id] || 0;
              const isProc = film.id === 'aruanda';
              const hasSel = film.id === r.film;
              const pct = cnt > 0 ? (cnt / maxM * 100) : 0;
              return (
                <div key={film.id}
                     className={'f-prj' + (hasSel ? ' active has-sel' : '') + (isProc ? ' proc' : '')}>
                  <span className="arr">{hasSel ? '▾' : '▸'}</span>
                  <span className="ico">{isProc ? '⟳' : '▣'}</span>
                  <span className="name">{film.title}</span>
                  <span className="ct">{film.scenes}</span>
                  <div className="meta">
                    <span style={{fontFamily:'var(--mono)', minWidth:30}}>{film.year}</span>
                    <span className="progress" style={{'--p': pct + '%'}}></span>
                    <span style={{fontFamily:'var(--mono)'}}>{cnt > 0 ? cnt + '/9' : '—'}</span>
                  </div>
                  {hasSel && (
                    <div className="selptr">
                      <span className="thumb" style={{backgroundImage:`url(${r.kf})`}}></span>
                      <span>cena {String(r.cena).padStart(3,'0')} · {r.tc}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="sect" style={{marginTop:4}}><span>Vistas salvas</span><button className="add">+</button></div>
          <div className="f-views">
            <div className="f-view"><span className="ico">◐</span><span>Exteriores rurais</span><span className="ct">142</span></div>
            <div className="f-view"><span className="ico">◐</span><span>Cartelas de título</span><span className="ct">28</span></div>
            <div className="f-view"><span className="ico">◐</span><span>Cenas noturnas</span><span className="ct">73</span></div>
            <div className="f-view"><span className="ico">◐</span><span>Pessoas em duo</span><span className="ct">96</span></div>
          </div>

          <div className="lp-foot">
            <div className="left">
              <span className="dot"></span>
              <span>Índice ok</span>
            </div>
            <div className="right">1.588 cenas · 8h54m</div>
          </div>
        </aside>

        {/* CENTER */}
        <section className="f-cp">
          <div className="f-search">
            <div className="f-srow">
              <span className="ico">⌕</span>
              <input defaultValue="duas pessoas conversando ao ar livre" />
              <span className="kbd">⌘K</span>
              <button className="submit">Buscar ⏎</button>
            </div>
            <div className="f-modes">
              <span className="f-chip on"><span className="ico">▤</span>Texto</span>
              <span className="f-chip"><span className="ico">⌬</span>Imagem</span>
              <span className="f-chip"><span className="ico">∿</span>Trilha</span>
              <span className="f-chip"><span className="ico">⊕</span>Multimodal</span>
              <span className="div"></span>
              <span className="f-knob"><span className="k">Híbrido</span><span className="v">sem 0.70 · bm25 0.30</span></span>
              <span className="f-knob"><span className="k">Rerank</span><span className="v acc">on</span></span>
              <span className="f-knob"><span className="k">MMR</span><span className="v">λ 0.50</span></span>
              <span className="f-knob"><span className="k">k</span><span className="v">9</span></span>
            </div>
          </div>

          <div className="f-caption">
            <div className="left">
              <span style={{fontWeight:600}}>9 cenas em <b>6 filmes</b></span>
              <span className="meta">
                <span>·</span>
                <b>231 ms</b>
                <span>·</span>
                <span>sem ⊕ bm25 ⊕ rerank</span>
              </span>
            </div>
            <div className="right">
              <span className="seg-tab on">⊞ Grade</span>
              <span className="seg-tab">≡ Lista</span>
              <span className="seg-tab">⊟ Compacto</span>
            </div>
          </div>

          <div className="f-grid">
            {results.map((rr, i) => {
              const ff = byId[rr.film];
              return (
                <article key={rr.id}
                         className={'f-card' + (i === sel ? ' sel' : '')}
                         onClick={() => setSel(i)}>
                  <div className="kf" style={{backgroundImage:`url(${rr.kf})`}}>
                    <span className="tag-tl"><span className="dot"></span>indexado</span>
                    <span className="tag-bl">{rr.tc}</span>
                    <span className="tag-tr">{rr.score.toFixed(3)}</span>
                    {(i === 0 || i === 3) && <span className="ann"><span className="dot"></span>1 anotação</span>}
                  </div>
                  <div className="body">
                    <div className="title">{rr.desc.split(',')[0].slice(0, 64)}…</div>
                    <div className="sub">
                      <span className="film-pill">{ff.title}</span>
                      <span className="yr">{ff.year} · cena {String(rr.cena).padStart(3,'0')}</span>
                    </div>
                    <p className="desc">{rr.desc}</p>
                    <div className="footrow">
                      <div className="tags">
                        {rr.tags.slice(0,3).map((t,j) => (
                          <span key={j} className={'tag' + (t==='duas-pessoas'||t==='exterior' ? ' m' : '')}>{t}</span>
                        ))}
                      </div>
                      <div className="acts">
                        <span className="a">⋯</span>
                      </div>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          {/* Bottom timeline strip — shows the SELECTED film's scene positions */}
          <div className="f-timeline">
            <div className="head">
              <div className="ttitle">
                <span>Timeline · {f.title}</span>
                <span className="pill">{f.scenes} cenas</span>
                <span className="pill" style={{color:F_PALETTE.accent}}>{matches[r.film]} matches</span>
              </div>
              <div className="controls">
                <span className="btn">⤡</span>
                <span>00:00:00</span>
                <span className="tc">{r.tc}</span>
                <span>{Math.floor(f.runtime/60)}:{String(f.runtime%60).padStart(2,'0')}:00</span>
                <span className="btn">⏵</span>
              </div>
            </div>
            <div className="scrubrow">
              <div className="scrub">
                {timelineSegs.map((kf, i) => {
                  const isSel = i === 11;
                  const isMatch = matchedIdx.includes(i);
                  return (
                    <span key={i} className={'seg' + (isMatch ? ' match' : '') + (isSel ? ' sel' : '')}
                          style={{backgroundImage:`url(${kf})`}}></span>
                  );
                })}
              </div>
              <div className="ticks">
                <span>00:00</span><span>12:00</span><span>24:00</span><span>36:00</span>
                <span>48:00</span><span>60:00</span><span>72:00</span><span>96:00</span>
              </div>
            </div>
          </div>
        </section>

        {/* RIGHT — Activity / Inspector */}
        <aside className="f-rp">
          <div className="htabs">
            <span className="tab on">Atividade<span className="pip">4</span></span>
            <span className="tab">Anotações<span className="pip">1</span></span>
            <span className="tab">Versões</span>
            <span className="gap"></span>
            <span className="iconbtn">⤡</span>
            <span className="iconbtn">⋯</span>
          </div>
          <div className="inner">
            <div className="f-insp-kf" style={{backgroundImage:`url(${r.kf})`}}>
              <div className="pin">1</div>
              <span className="pc"><span className="dot"></span>1 anotação · {r.tc}</span>
            </div>
            <div className="f-insp-meta">
              <div className="left">
                <h3>Cena {String(r.cena).padStart(3,'0')} · {f.title}</h3>
                <div className="at">
                  <span className="film-pill">{f.title}</span>
                  <span>{f.year}</span>
                  <span>·</span>
                  <span className="tc">{r.tc}</span>
                  <span>·</span>
                  <span>{f.director}</span>
                </div>
              </div>
              <div className="status"><span className="dot"></span>Indexado</div>
            </div>

            <div className="f-thread">
              <div className="f-comment ai">
                <div className="av">md</div>
                <div className="bx">
                  <div className="who">
                    <span className="name">moondream-2</span>
                    <span className="badge">AI · descrição</span>
                    <span className="when">há 4 dias</span>
                  </div>
                  <div className="body">"{r.desc}"</div>
                  <div className="replyrow">
                    <a>Editar</a>
                    <a>Marcar útil</a>
                    <a>Re-gerar</a>
                  </div>
                </div>
              </div>

              <div className="f-comment pinned curator">
                <div className="av">RG</div>
                <div className="bx">
                  <div className="who">
                    <span className="name">Rafael · curador</span>
                    <span className="badge">📍 fixado · {r.tc}</span>
                    <span className="when">há 2h</span>
                  </div>
                  <div className="body">Cena de referência para a vertente "diálogos no campo aberto". Boa candidata pro corte da retrospectiva 2026.</div>
                  <div className="replyrow">
                    <a>Responder</a>
                    <a>Resolver</a>
                  </div>
                </div>
              </div>
            </div>

            <div className="f-signals-card">
              <div className="h">
                <span>Por que este resultado</span>
                <span className="v">{r.score.toFixed(3)}</span>
              </div>
              <div className="row"><span className="lab">Semântico</span><span className="track" style={{'--p': `${(sigSem*100).toFixed(0)}%`}}></span><span className="v">{sigSem.toFixed(3)}</span></div>
              <div className="row"><span className="lab">BM25</span><span className="track" style={{'--p': `${(sigBm25*100).toFixed(0)}%`}}></span><span className="v">{sigBm25.toFixed(3)}</span></div>
              <div className="row"><span className="lab">Rerank</span><span className="track" style={{'--p': `${(sigRk*100).toFixed(0)}%`}}></span><span className="v">{sigRk.toFixed(3)}</span></div>
              <div className="row fused"><span className="lab">Fundido</span><span className="track" style={{'--p': `${(sigFu*100).toFixed(0)}%`}}></span><span className="v">{sigFu.toFixed(3)}</span></div>
            </div>

            <div className="f-rimas-card">
              <div className="h"><span>Rimas visuais · cross-film</span><a>Ver todas →</a></div>
              <div className="grid3">
                {rhymes.map((x, i) => {
                  const ff = byId[x.film];
                  return (
                    <div key={i} className="ry">
                      <div className="kf" style={{backgroundImage:`url(${x.kf})`}}></div>
                      <span className="lab">{ff.title} · <b>{x.sim}</b></span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{fontSize:11, color: F_PALETTE.muted, marginTop: 16, fontWeight: 500}}>TAGS</div>
            <div className="f-tagrow">
              {r.tags.map((t,i) => (
                <span key={i} className={'t' + (t==='duas-pessoas'||t==='exterior' ? ' m' : '')}>{t}</span>
              ))}
              <span className="add">+ Adicionar</span>
            </div>

            <div className="f-comment-input">
              <textarea placeholder="Adicionar comentário ou anotação…"></textarea>
              <div className="row">
                <div className="tools">
                  <span className="tool" title="Pin">📍</span>
                  <span className="tool" title="Tag">＃</span>
                  <span className="tool" title="Attach">⌘</span>
                  <span className="tool" title="Emoji">☺</span>
                </div>
                <button className="post">Comentar</button>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

window.MojicaFrame = MojicaFrame;
