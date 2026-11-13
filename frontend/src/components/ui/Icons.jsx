/**
 * Shared stroke icons (24x24, inherit currentColor) and the NeuroScope logo.
 * Used by the landing page and the app so both speak the same visual language.
 */
const PATHS = {
  upload: <path d="M12 16V4m0 0-4 4m4-4 4 4M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" />,
  cloud: <path d="M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 8.5a4 4 0 0 1 .5 7.97M12 12v8m0-8-3 3m3-3 3 3" />,
  database: <><ellipse cx="12" cy="5.5" rx="7" ry="2.5" /><path d="M5 5.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6M5 11.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6" /></>,
  detect: <><circle cx="11" cy="11" r="6" /><path d="m20 20-4.2-4.2M8.5 11h5M11 8.5v5" /></>,
  choose: <path d="M4 6h10M4 12h16M4 18h7M17 4v4M20 10v4M14 16v4" />,
  filter: <path d="M4 5h16l-6 7.5V19l-4-2v-4.5z" />,
  classify: <><circle cx="6" cy="6" r="2.2" /><circle cx="18" cy="6" r="2.2" /><circle cx="12" cy="18" r="2.2" /><path d="M7.6 7.6 10.6 16M16.4 7.6 13.4 16M8.2 6h7.6" /></>,
  results: <path d="M5 20V10m7 10V4m7 16v-7" />,
  wave: <path d="M2 12h4l2-6 4 12 3-9 2 3h5" />,
  shield: <path d="M12 3 5 6v6c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6z M9 12l2 2 4-4" />,
  browser: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18" /></>,
  layers: <path d="m12 3 9 5-9 5-9-5zm-9 9 9 5 9-5m-18 4 9 5 9-5" />,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  file: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>,
  zip: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M10 6h1M10 9h1M10 12h1M10 15h1.5v2.5H10z" /></>,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  alert: <><path d="M12 4 2.5 20h19z" /><path d="M12 10v4.5M12 17.5v.01" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8v.01" /></>,
  arrowLeft: <path d="M19 12H5m0 0 6-6m-6 6 6 6" />,
  arrowRight: <path d="M5 12h14m0 0-6-6m6 6-6 6" />,
  external: <path d="M14 4h6v6M20 4l-9 9M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />,
  user: <><circle cx="12" cy="8" r="4" /><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  bolt: <path d="M13 3 5 14h6l-1 7 8-11h-6z" />,
  sparkle: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
};

export function Icon({ name, className = "", size }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
         aria-hidden="true">
      {PATHS[name]}
    </svg>
  );
}

export function Logo({ className = "" }) {
  return (
    <svg className={className} viewBox="0 0 32 24" fill="none" stroke="currentColor" strokeWidth="2.4"
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M1 13h6l3-9 5 16 4-12 3 5h9" />
    </svg>
  );
}
