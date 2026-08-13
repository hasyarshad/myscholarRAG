import type { NextConfig } from "next";

const config: NextConfig = {
  // The API base is server-side only, so the browser never learns where the
  // backend lives. Phase 7 puts a real proxy here; for local development the
  // widget talks to it directly.
  env: { MYSCHOLAR_API: process.env.MYSCHOLAR_API ?? "http://127.0.0.1:8000" },
};

export default config;
