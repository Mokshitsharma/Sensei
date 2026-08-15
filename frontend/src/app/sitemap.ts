import type { MetadataRoute } from "next";
import { api } from "@/lib/api";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const base: MetadataRoute.Sitemap = [
    { url: SITE_URL, changeFrequency: "hourly", priority: 1 },
  ];

  try {
    const stocks = await api.stocks();
    const stockEntries: MetadataRoute.Sitemap = stocks.map((s) => ({
      url: `${SITE_URL}/stock/${encodeURIComponent(s.ticker)}`,
      changeFrequency: "hourly",
      priority: 0.7,
    }));
    return [...base, ...stockEntries];
  } catch {
    return base;
  }
}
