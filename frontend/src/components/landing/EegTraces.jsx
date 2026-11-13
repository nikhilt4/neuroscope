import { useEffect, useRef } from "react";

/**
 * Decorative scrolling EEG for the landing hero.
 *
 * Real scalp EEG has a roughly 1/f power spectrum (more power at low frequencies),
 * waxing-and-waning alpha bursts, and occasional sharp transients. This sums ~18
 * sinusoids with 1/f-scaled amplitudes and random phases per channel, modulates a
 * ~10 Hz component, and adds rare Gaussian "spikes". It looks like EEG; it is not EEG.
 *
 * The labelled, event-driven version (cues + mu suppression) is reserved for the
 * live-simulation feature, where it can be driven by real data.
 */

const TAU = Math.PI * 2;

// deterministic pseudo-random so every load draws the same waveform
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeChannel(seed) {
  const rand = mulberry32(seed);
  const comps = Array.from({ length: 18 }, (_, k) => {
    const f = 0.6 * Math.pow(1.24, k);                 // 0.6 Hz .. ~23 Hz, log-spaced
    return { f, a: 1 / Math.pow(f, 0.85), p: rand() * TAU };
  });
  const norm = comps.reduce((s, c) => s + c.a, 0) / 2.2;
  const spikes = Array.from({ length: 5 }, () => ({ t: rand() * 20, a: (rand() - 0.5) * 2.2 }));
  return { comps, norm, spikes, alphaGain: 0.25 + rand() * 0.55, alphaPhase: rand() * TAU };
}

function value(ch, t) {
  let v = 0;
  for (const c of ch.comps) v += c.a * Math.sin(TAU * c.f * t + c.p);
  v /= ch.norm;
  const burst = 0.5 + 0.5 * Math.sin(0.45 * t + ch.alphaPhase);
  v += ch.alphaGain * burst * burst * Math.sin(TAU * 10.3 * t + ch.alphaPhase);
  const tm = ((t % 20) + 20) % 20;                      // spikes repeat every 20 s
  for (const s of ch.spikes) {
    const d = tm - s.t;
    v += s.a * Math.exp(-(d * d) / 0.0012);
  }
  return v;
}

function EegTraces({ channels = 6, color = "#6d8cf5", windowS = 5, speed = 0.6, className = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const chans = Array.from({ length: channels }, (_, i) => makeChannel(1234 + i * 97));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, raf = 0, visible = true;
    const start = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      w = r.width; h = r.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now) => {
      const t1 = reduceMotion ? windowS : windowS + ((now - start) / 1000) * speed;
      const t0 = t1 - windowS;
      const rowH = h / channels;
      ctx.clearRect(0, 0, w, h);
      ctx.lineWidth = 1.6;
      ctx.lineJoin = "round";
      chans.forEach((ch, i) => {
        const y0 = rowH * (i + 0.5);
        const amp = rowH * 0.32;
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45 + 0.55 * (i / Math.max(1, channels - 1));
        ctx.beginPath();
        for (let x = 0; x <= w; x += 1.5) {
          const y = y0 - value(ch, t0 + (x / w) * windowS) * amp;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      if (!reduceMotion && visible) raf = requestAnimationFrame(draw);
    };

    resize();
    const ro = new ResizeObserver(() => { resize(); if (reduceMotion) draw(0); });
    ro.observe(canvas);
    // pause while off-screen
    const io = new IntersectionObserver(([e]) => {
      const was = visible;
      visible = e.isIntersecting;
      if (visible && !was && !reduceMotion) raf = requestAnimationFrame(draw);
    });
    io.observe(canvas);
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); };
  }, [channels, color, windowS, speed]);

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
}

export default EegTraces;
