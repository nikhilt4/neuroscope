import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../../api";
import { Icon } from "../ui/Icons";
import { prettyLabel } from "../../labels";
import "./Playback.css";

// Clinical-style EEG page viewer. Replaces the earlier video-style replay after a
// neuroscientist found it slow and asked for a scrollable sheet like a clinical EEG
// reader: every channel stacked, one page of time at a time, events marked on the page,
// and a click on any event jumps straight to it.

const CHUNK_S = 10;                         // the backend is asked for fixed 10-s chunks, cached here
const MAX_CHUNKS = 48;                      // ~8 min of cached data per montage
const PAGE_LENGTHS = [5, 10, 20, 30];       // 10 s per page is the clinical default
const SENSITIVITIES = [10, 20, 30, 50, 70, 100, 150, 200, 300, 500];  // µV from lane centre to edge
const SPEEDS = [1, 2, 5, 10];
const LABEL_W = 84;
const GROUP_COLORS = {
  left: "#2563eb", right: "#dc2626", midline: "#1e293b",   // as in clinical displays: left blue, right red
  ecg: "#c026d3", eog: "#0f9f6e",
};
const EVENT_COLORS = ["#4f46e5", "#ec4899", "#0ea5e9", "#8b5cf6", "#22a565", "#f59e0b", "#ef4444"];

function fmtTime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

function PlaybackView({ sessionId, title, onClose, labelNames = {} }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [montage, setMontage] = useState("referential");
  const [pageLen, setPageLen] = useState(10);
  const [sens, setSens] = useState(100);   // µV per half lane; 50 crowded real recordings
  const [start, setStart] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [hidden, setHidden] = useState(() => new Set());
  const [eventFilter, setEventFilter] = useState(null);   // show one label in the event list
  const [chunkTick, setChunkTick] = useState(0);          // bumps when a chunk arrives → redraw

  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const chunks = useRef(new Map());                       // `${montage}|${k}` → page from the backend
  const pending = useRef(new Set());
  const startRef = useRef(0);
  startRef.current = start;

  const label = useCallback((l) => labelNames[l] || prettyLabel(l), [labelNames]);

  useEffect(() => {
    api.getPlaybackInfo(sessionId)
      .then((i) => { setInfo(i); setMontage(i.montages?.includes("bipolar") ? "bipolar" : "referential"); })
      .catch((e) => setError(e.message));
  }, [sessionId]);

  const duration = info?.duration_s ?? 0;
  const colorOf = useMemo(() => {
    const out = {};
    (info?.event_labels || []).forEach((l, k) => { out[l] = EVENT_COLORS[k % EVENT_COLORS.length]; });
    return out;
  }, [info]);
  const clampStart = useCallback((s) => Math.min(Math.max(0, s), Math.max(0, duration - pageLen)), [duration, pageLen]);

  // ── data: fetch the chunks covering the page (plus the next one, so Play doesn't stall)
  useEffect(() => {
    if (!info) return;
    const first = Math.floor(start / CHUNK_S);
    const last = Math.floor((start + pageLen) / CHUNK_S) + 1;
    for (let k = first; k <= last; k++) {
      if (k * CHUNK_S >= duration) break;
      const key = `${montage}|${k}`;
      if (chunks.current.has(key) || pending.current.has(key)) continue;
      pending.current.add(key);
      api.getPlaybackWindow(sessionId, { start: k * CHUNK_S, duration: CHUNK_S, montage })
        .then((page) => {
          chunks.current.set(key, page);
          while (chunks.current.size > MAX_CHUNKS) chunks.current.delete(chunks.current.keys().next().value);
          setChunkTick((n) => n + 1);
        })
        .catch((e) => { setError(e.message); setPlaying(false); })
        .finally(() => pending.current.delete(key));
    }
  }, [info, start, pageLen, montage, sessionId, duration]);

  // channel list for this montage comes from the first chunk that arrives
  const channelMeta = useMemo(() => {
    for (const [key, page] of chunks.current) if (key.startsWith(`${montage}|`)) return page.channels;
    return [];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [montage, chunkTick]);
  const shown = channelMeta.map((c, i) => ({ ...c, i })).filter((c) => !hidden.has(c.name));
  const rowH = shown.length ? Math.max(26, Math.min(56, Math.floor(620 / shown.length))) : 40;
  const height = Math.max(240, shown.length * rowH + 28);

  // ── draw the page
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !info) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    canvas.width = w * dpr; canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, height);
    const plotW = w - LABEL_W - 8;
    const x = (t) => LABEL_W + ((t - start) / pageLen) * plotW;
    const top = 22;

    // 1-s grid, stronger every 5 s, with time labels along the top
    for (let s = Math.ceil(start); s <= start + pageLen; s++) {
      const gx = x(s);
      ctx.strokeStyle = s % 5 === 0 ? "rgba(14,165,233,.45)" : "rgba(14,165,233,.18)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(gx, top); ctx.lineTo(gx, height); ctx.stroke();
      if (s % (pageLen > 10 ? 5 : 1) === 0) {
        ctx.fillStyle = "#8a93ad"; ctx.font = "500 10px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
        ctx.fillText(fmtTime(s), gx, 14);
      }
    }

    // traces, stitched from whichever cached chunks cover the page
    const pages = [...chunks.current.entries()].filter(([k]) => k.startsWith(`${montage}|`)).map(([, p]) => p)
      .filter((p) => p.start < start + pageLen && p.start + p.duration > start)
      .sort((a, b) => a.start - b.start);
    shown.forEach((ch, row) => {
      const mid = top + row * rowH + rowH / 2;
      const color = GROUP_COLORS[ch.group] || "#475569";
      ctx.fillStyle = color; ctx.font = "600 11px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
      ctx.fillText(ch.name, 6, mid + 4);
      ctx.save();
      ctx.beginPath(); ctx.rect(LABEL_W, mid - rowH, plotW, rowH * 2); ctx.clip();   // a trace may cross one lane, no more
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
      let pen = false;
      pages.forEach((p) => {
        const series = p.data[ch.i];
        if (!series) return;
        const i0 = Math.max(0, Math.floor((start - p.start) * p.sfreq));
        const i1 = Math.min(series.length, Math.ceil((start + pageLen - p.start) * p.sfreq) + 1);
        for (let i = i0; i < i1; i++) {
          const px = x(p.start + i / p.sfreq), py = mid - (series[i] / sens) * (rowH / 2);
          if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
        }
      });
      ctx.stroke();
      ctx.restore();
    });

    // events on the page: a line through every channel, the label at the top
    (info.events || []).forEach((e) => {
      if (e.t < start || e.t > start + pageLen) return;
      const ex = x(e.t);
      ctx.strokeStyle = colorOf[e.label]; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ex, top); ctx.lineTo(ex, height); ctx.stroke();
      ctx.fillStyle = colorOf[e.label]; ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif"; ctx.textAlign = "left";
      ctx.fillText(label(e.label), ex + 4, top + 12);
    });

    // scale bar in the header strip, right of the time labels: `sens` µV is half a lane
    const bx = 10, bh = Math.min(rowH / 2, top - 4);
    ctx.strokeStyle = "#4f46e5"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(bx, 3); ctx.lineTo(bx, 3 + bh); ctx.stroke();
    ctx.fillStyle = "#4f46e5"; ctx.font = "600 10px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    ctx.fillText(`${sens * (bh / (rowH / 2)) | 0} µV`, bx + 5, 12);

    if (!pages.length) {
      ctx.fillStyle = "#8a93ad"; ctx.font = "500 13px 'Plus Jakarta Sans', sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Loading…", w / 2, height / 2);
    }
  }, [info, start, pageLen, sens, montage, height, rowH, shown, chunkTick, colorOf, label]);

  // ── Play = auto-scroll the page
  useEffect(() => {
    if (!playing) return;
    let raf, last = performance.now();
    const step = (now) => {
      const next = startRef.current + ((now - last) / 1000) * speed;
      last = now;
      if (next >= duration - pageLen) { setStart(clampStart(next)); setPlaying(false); return; }
      setStart(next);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration, pageLen, clampStart]);

  // ── mouse wheel, drag and keys
  const go = useCallback((s) => setStart(clampStart(s)), [clampStart]);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // non-passive so the page itself doesn't scroll while you scroll through the recording
    const onWheel = (e) => { e.preventDefault(); go(startRef.current + (e.deltaY + e.deltaX) / 100 * pageLen * 0.1); };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [go, pageLen, info]);
  const drag = useRef(null);
  const onPointerDown = (e) => { drag.current = { x: e.clientX, s: start }; e.currentTarget.setPointerCapture(e.pointerId); };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const plotW = (canvasRef.current?.clientWidth || 1000) - LABEL_W - 8;
    go(drag.current.s - ((e.clientX - drag.current.x) / plotW) * pageLen);
  };
  const onPointerUp = () => { drag.current = null; };
  const onKeyDown = (e) => {
    const page = e.shiftKey || e.key === "PageDown" || e.key === "PageUp";
    const map = { ArrowRight: 1, ArrowLeft: -1, PageDown: 1, PageUp: -1 };
    if (map[e.key]) { e.preventDefault(); go(start + map[e.key] * (page ? pageLen : 1)); }
    if (e.key === "Home") go(0);
    if (e.key === "End") go(duration);
    if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
  };

  // events: jump so the event sits a quarter of the way into the page
  const jumpTo = (t) => { setPlaying(false); go(t - pageLen / 4); };
  const listed = (info?.events || []).filter((e) => !eventFilter || e.label === eventFilter);
  const nextEvent = (dir) => {
    const ref = start + pageLen / 4 + dir * 0.01;
    const evs = info?.events || [];
    const hit = dir > 0 ? evs.find((e) => e.t > ref) : [...evs].reverse().find((e) => e.t < ref);
    if (hit) jumpTo(hit.t);
  };

  const toggleChannel = (name) => setHidden((prev) => {
    const next = new Set(prev);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });

  if (error && !info) {
    return (
      <div className="pb">
        <button className="up-link" onClick={onClose}><Icon name="arrowLeft" />Back to the analysis</button>
        <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>
      </div>
    );
  }
  if (!info) return <p className="ch-empty">Loading the recording…</p>;

  const [hp, lp] = info.display_filter_hz || [];
  const MONTAGE_LABEL = { referential: "Referential", average: "Average reference", bipolar: "Bipolar (double banana)" };

  return (
    <div className="pb">
      <div className="pb-head">
        <button className="up-link" onClick={() => { setPlaying(false); onClose(); }}><Icon name="arrowLeft" />Back to the analysis</button>
        <h2>{title}</h2>
        <p>
          {info.sfreq} Hz · {fmtTime(duration)} long · display filter {hp}–{lp} Hz
          {/* the online demo ships an excerpt, not the whole night */}
          {info.clip_start_s != null && <> · excerpt from {fmtTime(info.clip_start_s)} into a {fmtTime(info.recording_duration_s)} recording</>}
        </p>
      </div>

      <div className="pb-toolbar">
        <label>Montage
          <select value={montage} onChange={(e) => setMontage(e.target.value)}>
            {(info.montages || ["referential"]).map((m) => <option key={m} value={m}>{MONTAGE_LABEL[m] || m}</option>)}
          </select>
        </label>
        <label>Page
          <select value={pageLen} onChange={(e) => { const v = Number(e.target.value); setPageLen(v); }}>
            {PAGE_LENGTHS.map((p) => <option key={p} value={p}>{p} s</option>)}
          </select>
        </label>
        <div className="pb-seg" role="group" aria-label="Sensitivity">
          <button onClick={() => setSens(SENSITIVITIES[Math.max(0, SENSITIVITIES.indexOf(sens) - 1)])} aria-label="Larger traces">+</button>
          <span>{sens} µV</span>
          <button onClick={() => setSens(SENSITIVITIES[Math.min(SENSITIVITIES.length - 1, SENSITIVITIES.indexOf(sens) + 1)])} aria-label="Smaller traces">−</button>
        </div>
        <span className="pb-pagetime">{fmtTime(start)} – {fmtTime(Math.min(duration, start + pageLen))}</span>
      </div>

      <div className="pb-screen" ref={wrapRef} tabIndex={0} onKeyDown={onKeyDown}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
           aria-label="EEG page. Scroll, drag or use the arrow keys to move through the recording.">
        <canvas ref={canvasRef} className="pb-canvas" style={{ height }} />
      </div>

      <div className="pb-controls">
        <div className="pb-nav" role="group" aria-label="Move through the recording">
          <button onClick={() => go(0)} title="Start (Home)">|◀</button>
          <button onClick={() => go(start - pageLen)} title="Previous page (Shift+←)">◀◀</button>
          <button onClick={() => go(start - 1)} title="Back 1 s (←)">◀</button>
          <button onClick={() => go(start + 1)} title="Forward 1 s (→)">▶</button>
          <button onClick={() => go(start + pageLen)} title="Next page (Shift+→)">▶▶</button>
          <button onClick={() => go(duration)} title="End (End)">▶|</button>
        </div>
        <button className="wz-btn pb-play" onClick={() => setPlaying((p) => !p)}>
          {playing ? <><svg viewBox="0 0 24 24" width="16"><path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" /></svg>Pause</>
                   : <><svg viewBox="0 0 24 24" width="16"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>Play</>}
        </button>
        <div className="pb-seg" role="group" aria-label="Play speed">
          {SPEEDS.map((s) => <button key={s} className={s === speed ? "is-active" : ""} onClick={() => setSpeed(s)}>{s}×</button>)}
        </div>
        <div className="pb-timeline">
          <div className="pb-ticks" aria-hidden="true">
            {info.events.length <= 2000 && info.events.map((e, i) => (
              <i key={i} style={{ left: `${(e.t / duration) * 100}%`, background: colorOf[e.label] }} />
            ))}
          </div>
          <input type="range" min={0} max={Math.max(0, duration - pageLen)} step={0.1} value={start} aria-label="Position"
                 onChange={(e) => go(Number(e.target.value))} />
        </div>
      </div>

      {error && <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>}

      <div className="pb-panels">
        <section className="pb-card">
          <h4>
            Events <span>{info.events.length} in this recording</span>
            {info.events.length > 0 && (
              <span className="pb-evnav">
                <button onClick={() => nextEvent(-1)}>← Previous</button>
                <button onClick={() => nextEvent(1)}>Next →</button>
              </span>
            )}
          </h4>
          <ul className="pb-legend">
            {info.event_labels.map((l) => (
              <li key={l}>
                <button className={eventFilter === l ? "is-on" : ""} onClick={() => setEventFilter(eventFilter === l ? null : l)}>
                  <i style={{ background: colorOf[l] }} />{label(l)}
                  <small>{info.events.filter((e) => e.label === l).length}</small>
                </button>
              </li>
            ))}
          </ul>
          {listed.length > 0 && (
            <ol className="pb-evlist">
              {listed.slice(0, 500).map((e, i) => (
                <li key={`${e.t}-${i}`}>
                  <button onClick={() => jumpTo(e.t)} className={e.t >= start && e.t <= start + pageLen ? "is-here" : ""}>
                    <span>{fmtTime(e.t)}</span><b style={{ color: colorOf[e.label] }}>{label(e.label)}</b>
                  </button>
                </li>
              ))}
            </ol>
          )}
          {info.events.length === 0 && <p className="config-hint">This recording has no event markers.</p>}
        </section>
        <section className="pb-card">
          <h4>Channels <span>{shown.length} of {channelMeta.length} shown · click to hide or show</span></h4>
          <div className="pb-chips">
            {channelMeta.map((c) => (
              <button key={c.name} className={hidden.has(c.name) ? "" : "is-on"} onClick={() => toggleChannel(c.name)}
                      style={hidden.has(c.name) ? undefined : { background: GROUP_COLORS[c.group] || "#475569", borderColor: "transparent" }}>
                {c.name}
              </button>
            ))}
          </div>
          <p className="config-hint pb-keys">Scroll or drag the page to move through time · ← → 1 s · Shift+← → one page · Space plays</p>
        </section>
      </div>
    </div>
  );
}

export default PlaybackView;
