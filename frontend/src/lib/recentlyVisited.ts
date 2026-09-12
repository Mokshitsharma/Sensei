const KEY = "sensei-recent";
const MAX_ENTRIES = 10;

export type RecentEntry = {
  ticker: string;
  name: string;
  visitedAt: number;
};

export function getRecentlyVisited(): RecentEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function pushRecent(entry: { ticker: string; name: string }) {
  if (typeof window === "undefined") return;
  const existing = getRecentlyVisited().filter((e) => e.ticker !== entry.ticker);
  const next = [{ ...entry, visitedAt: Date.now() }, ...existing].slice(0, MAX_ENTRIES);
  localStorage.setItem(KEY, JSON.stringify(next));
}
