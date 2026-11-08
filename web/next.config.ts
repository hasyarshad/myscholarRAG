import type { NextConfig } from "next";

const config: NextConfig = {
  // The dev server serves its static chunks only to `localhost` by default. From
  // any other host the page renders but never hydrates, so the Ask button stays
  // disabled. Development only.
  allowedDevOrigins: ["127.0.0.1"],
};

export default config;
