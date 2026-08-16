import type { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

// /explore and /stock/* require sign-in (see middleware.ts) — crawlers
// can't access them anyway, so only the public landing page is listed.
export default function sitemap(): MetadataRoute.Sitemap {
  return [{ url: SITE_URL, changeFrequency: "hourly", priority: 1 }];
}
