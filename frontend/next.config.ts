import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Browser automation/tooling in this dev environment loads the app via
  // 127.0.0.1 rather than localhost — Next 16's allowedDevOrigins check
  // otherwise silently blocks JS chunk requests from that origin, which
  // breaks client hydration app-wide with no console error (SSR HTML
  // renders fine, but no client interactivity works at all).
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
