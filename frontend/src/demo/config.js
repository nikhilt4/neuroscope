// The online demo is a static build (npm run build:demo) with no backend: every API call
// is answered from recordings in public/demo/. VITE_DEMO comes from .env.demo, and Vite
// replaces it at build time, so the normal build drops all the demo code.
// https://vite.dev/guide/env-and-mode (Vite 5)
export const DEMO = import.meta.env.VITE_DEMO === "1";

// Link shown in the demo notes. Empty hides it — set VITE_REPO_URL in .env.demo once the repo is public.
export const REPO_URL = import.meta.env.VITE_REPO_URL || "";
