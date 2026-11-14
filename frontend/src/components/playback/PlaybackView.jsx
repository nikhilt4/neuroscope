import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../../api";
import { Icon } from "../ui/Icons";
import { prettyClass } from "../results/metrics";
import "./Playback.css";

const WINDOW_S = 10;                       // seconds visible across the screen
const SPEEDS = [0.5, 1, 2, 4];
const GAINS = [10, 20, 50, 100, 200, 500]; // µV from a lane's centre line to its edge
const EVENT_COLORS = ["#4f46e5", "#ec4899", "#0ea5e9", "#8b5cf6", "#22a565", "#f59e0b", "#ef4444"];
const LABEL_W = 56;

function fmtTime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

/**
 * Real-time replay of a continuous public recording: scrolling traces + event markers.
 * The server streams 1/25 s frames (Server-Sent Events); this component keeps the last
 * WINDOW_S seconds in a ring buffer per channel and redraws on every animation frame.
 * Pause / seek / speed / channel changes close the stream and reopen it at the playhead.
 */
function PlaybackView({ sessionId, title, onClose }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [channels, setChannels] = useState([]);
  const [speed, setSpeed] = useState(1);
  const [gain, setGain] = useState(50);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [recent, setRecent] = useState([]);   // last few events, newest first

  const canvasRef = useRef(null);
  const closeRef = useRef(null);
  const buf = useRef(null);                     // { data: Float32Array[], head, sfreq, n }
  const evRef = useRef([]);                     // events inside the visible window
  const tRef = useRef(0);
  const colorOf = useRef({});

  useEffect(() => {
    api.getPlaybackInfo(sessionId)
      .then((i) => {
        setInfo(i);
        setChannels(i.default_channels);
        i.event_labels.forEach((l, k) => { colorOf.current[l] = EVENT_COLORS[k % EVENT_COLORS.length]; });
      })
      .catch((e) => setError(e.message));
    return () => closeRef.current?.();
  }, [sessionId]);

  const resetBuffer = useCallback((sfreq, nCh) => {
    const n = Math.round(WINDOW_S * sfreq);
    buf.current = { data: Array.from({ length: nCh }, () => new Float32Array(n).fill(NaN)), head: 0, sfreq, n };
    evRef.current = [];
  }, []);

  const stop = useCallback(() => { closeRef.current?.(); closeRef.current = null; setPlaying(false); }, []);

  // Opens a stream at `from`. Every open asks for the preceding WINDOW_S seconds as a
  // prefill, so the screen is full immediately (like scrubbing a video) instead of
  // refilling from blank. opts.preview: just draw that window and stop (paused scrubbing).
  const start = useCallback((from, opts = {}) => {
    if (!info) return;
    closeRef.current?.();
    const chans = opts.channels ?? channels;
    resetBuffer(info.sfreq, chans.length);
    setError(null);
    const preview = !!opts.preview;
    closeRef.current = api.openPlayback(sessionId, {
      start: from, speed: opts.speed ?? speed, channels: chans, prefill: WINDOW_S, preview,
    }, (msg) => {
      const b = buf.current;
      if (msg.type === "data") {
        const len = msg.samples[0].length;
        for (let c = 0; c < msg.samples.length; c++) {
          const ring = b.data[c], src = msg.samples[c];
          for (let i = 0; i < len; i++) ring[(b.head + i) % b.n] = src[i];
        }
        b.head = (b.head + len) % b.n;
        tRef.current = msg.t + len / b.sfreq;
      } else if (msg.type === "event") {
        evRef.current.push(msg);
        if (!msg.prefill) setRecent((r) => [msg, ...r].slice(0, 6));   // the log is "as it happens"
      } else if (msg.type === "end") {
        if (!msg.preview) setPlaying(false);
      }
    }, (e) => { setPlaying(false); setError(e.message); });
    tRef.current = from;
    setT(from);
    setPlaying(!preview);
  }, [info, channels, speed, sessionId, resetBuffer]);

  // Seek from the slider (drag, scroll wheel, arrow keys). Debounced so a fast scroll
  // opens one stream at the final position, not one per notch.
  const seekTimer = useRef(null);
  const playingRef = useRef(false);
  useEffect(() => { playingRef.current = playing; }, [playing]);
  const seekTo = useCallback((v) => {
    setT(v);
    tRef.current = v;
    clearTimeout(seekTimer.current);
    seekTimer.current = setTimeout(() => start(v, { preview: !playingRef.current }), 150);
  }, [start]);
  useEffect(() => () => clearTimeout(seekTimer.current), []);

  // keep the React playhead roughly in sync for the timeline (4×/s is plenty)
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => setT(tRef.current), 250);
    return () => clearInterval(id);
  }, [playing]);

  // draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !info) return undefined;
    const ctx = canvas.getContext("2d");
    let raf;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      if (canvas.width !== Math.round(width * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const b = buf.current;
      const plotW = width - LABEL_W;
      const lanes = Math.max(1, channels.length);
      const laneH = height / lanes;
      const tEnd = tRef.current, tStart = tEnd - WINDOW_S;

      // second gridlines
      ctx.strokeStyle = "#eef0f8"; ctx.lineWidth = 1;
      for (let s = Math.ceil(tStart); s <= tEnd; s++) {
        const x = LABEL_W + ((s - tStart) / WINDOW_S) * plotW;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      // event markers
      evRef.current = evRef.current.filter((e) => e.t >= tStart - 1);
      for (const e of evRef.current) {
        const x = LABEL_W + ((e.t - tStart) / WINDOW_S) * plotW;
        if (x < LABEL_W) continue;
        const col = colorOf.current[e.label] || "#4f46e5";
        ctx.strokeStyle = col; ctx.globalAlpha = 0.8; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(x, 18); ctx.lineTo(x, height); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        const text = prettyClass(e.label);
        ctx.font = "600 11px 'JetBrains Mono', monospace";
        const w = ctx.measureText(text).width + 12;
        ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(Math.min(x + 3, width - w - 2), 2, w, 16, 8); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.fillText(text, Math.min(x + 3, width - w - 2) + 6, 14);
      }
      // traces
      channels.forEach((ch, c) => {
        const mid = laneH * (c + 0.5);
        ctx.fillStyle = "#5e6987"; ctx.font = "600 11px 'JetBrains Mono', monospace";
        ctx.fillText(ch, 6, mid + 4);
        ctx.strokeStyle = "#f1f3fa"; ctx.beginPath(); ctx.moveTo(LABEL_W, mid); ctx.lineTo(width, mid); ctx.stroke();
        if (!b || !b.data[c]) return;
        const ring = b.data[c];
        ctx.strokeStyle = c % 2 ? "#6366f1" : "#3b5bdb"; ctx.lineWidth = 1.2;
        ctx.beginPath();
        let started = false;
        const step = Math.max(1, Math.floor(b.n / (plotW * 2)));   // ~2 points per pixel
        for (let i = 0; i < b.n; i += step) {
          const v = ring[(b.head + i) % b.n];
          if (Number.isNaN(v)) { started = false; continue; }
          const x = LABEL_W + (i / b.n) * plotW;
          const y = mid - Math.max(-1, Math.min(1, v / gain)) * (laneH / 2) * 0.9;
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [info, channels, gain]);

  if (error && !info) {
    return (
      <div className="pb">
        <button className="up-link" onClick={onClose}><Icon name="arrowLeft" />Back</button>
        <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>
      </div>
    );
  }
  if (!info) return <div className="pb"><p className="ch-empty">Loading the recording…</p></div>;

  const restartIfPlaying = (opts) => { if (playing) start(tRef.current, opts); };
  const toggleChannel = (ch) => {
    const next = channels.includes(ch) ? channels.filter((c) => c !== ch) : [...channels, ch].slice(-8);
    if (!next.length) return;
    setChannels(next);
    if (playing) start(tRef.current, { channels: next }); else resetBuffer(info.sfreq, next.length);
  };

  return (
    <div className="pb">
      <div className="pb-head">
        <button className="up-link" onClick={() => { stop(); onClose(); }}><Icon name="arrowLeft" />Back to the analysis</button>
        <h2>{title}</h2>
        <p>
          Replaying the recording in real time · {info.sfreq} Hz · {fmtTime(info.duration_s)} long ·
          display filter {info.display_filter_hz[0]}–{info.display_filter_hz[1]} Hz
        </p>
      </div>

      <div className="pb-screen">
        <canvas ref={canvasRef} className="pb-canvas" style={{ height: Math.max(260, channels.length * 62) }} />
        {!playing && t === 0 && (
          <button className="pb-big-play" onClick={() => start(0)} aria-label="Play">
            <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
          </button>
        )}
      </div>

      <div className="pb-controls">
        <button className="wz-btn pb-play" onClick={() => (playing ? stop() : start(t >= info.duration_s - 0.1 ? 0 : t))}>
          {playing ? <><svg viewBox="0 0 24 24" width="16"><path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" /></svg>Pause</>
                   : <><svg viewBox="0 0 24 24" width="16"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>Play</>}
        </button>
        <span className="pb-time">{fmtTime(t)} / {fmtTime(info.duration_s)}</span>
        <div className="pb-timeline">
          <div className="pb-ticks" aria-hidden="true">
            {info.events.length <= 2000 && info.events.map((e, i) => (
              <i key={i} style={{ left: `${(e.t / info.duration_s) * 100}%`, background: colorOf.current[e.label] }} />
            ))}
          </div>
          <input type="range" min={0} max={info.duration_s} step={0.1} value={t} aria-label="Seek"
                 onChange={(e) => seekTo(Number(e.target.value))} />
        </div>
        <div className="pb-seg" role="group" aria-label="Speed">
          {SPEEDS.map((s) => (
            <button key={s} className={s === speed ? "is-active" : ""}
                    onClick={() => { setSpeed(s); restartIfPlaying({ speed: s }); }}>{s}×</button>
          ))}
        </div>
        <div className="pb-seg" role="group" aria-label="Gain">
          <button onClick={() => setGain(GAINS[Math.max(0, GAINS.indexOf(gain) - 1)])} aria-label="Larger traces">+</button>
          <span>{gain} µV</span>
          <button onClick={() => setGain(GAINS[Math.min(GAINS.length - 1, GAINS.indexOf(gain) + 1)])} aria-label="Smaller traces">−</button>
        </div>
      </div>

      {error && <div className="wz-alert"><Icon name="alert" /><span>{error}</span></div>}

      <div className="pb-panels">
        <section className="pb-card">
          <h4>Events <span>{info.events.length} in this recording</span></h4>
          <ul className="pb-legend">
            {info.event_labels.map((l) => (
              <li key={l}><i style={{ background: colorOf.current[l] }} />{prettyClass(l)}
                <small>{info.events.filter((e) => e.label === l).length}</small></li>
            ))}
          </ul>
          {recent.length > 0 && (
            <ol className="pb-recent">
              {recent.map((e, i) => (
                <li key={`${e.t}-${i}`}><span>{fmtTime(e.t)}</span><b style={{ color: colorOf.current[e.label] }}>{prettyClass(e.label)}</b></li>
              ))}
            </ol>
          )}
        </section>
        <section className="pb-card">
          <h4>Channels <span>{channels.length} of {info.channels.length} shown · up to 8</span></h4>
          <div className="pb-chips">
            {info.channels.map((ch) => (
              <button key={ch} className={channels.includes(ch) ? "is-on" : ""} onClick={() => toggleChannel(ch)}>{ch}</button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

export default PlaybackView;
