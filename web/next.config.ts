import type { NextConfig } from "next";

const config: NextConfig = {
  // The API base is server-side only, so the browser never learns where the
  // backend lives. Phase 7 puts a real proxy here; for local development the
  // widget talks to it directly.
  env: { MYSCHOLAR_API: process.env.MYSCHOLAR_API ?? "http://127.0.0.1:8000" },

  // Next blocks cross-origin requests to dev-only assets, and treats the
  // hostname it was started with — `localhost` — as the only permitted origin.
  // Opening the same server on http://127.0.0.1:3100, or on the LAN address it
  // prints at startup, therefore returns 403 for two `/_next/static/chunks`
  // files.
  //
  // The failure is worth describing, because nothing about it says "wrong
  // hostname". The server-rendered HTML arrives intact, so the page looks
  // completely normal and the input accepts typing. What is missing is
  // hydration: React never attaches, so `onChange` never runs, `draft` stays
  // empty, and the Ask button — `disabled={!draft.trim()}` — is greyed for
  // ever. The starter-question chips disappear at the same time, because they
  // render from a `corpus` fetch inside an effect that also never runs. It
  // presents as "the widget won't let me ask anything" with a healthy API, a
  // healthy proxy, and every server-side check passing.
  //
  // Only development is affected. `next build` emits static chunks with no
  // origin check, so production has never had this.
  allowedDevOrigins: ["127.0.0.1"],
};

export default config;
