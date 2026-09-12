"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getRecentlyVisited, type RecentEntry } from "@/lib/recentlyVisited";

export function RecentlyVisitedRow() {
  const [entries, setEntries] = useState<RecentEntry[] | null>(null);

  useEffect(() => {
    setEntries(getRecentlyVisited());
  }, []);

  if (entries === null || entries.length === 0) return null;

  return (
    <div className="mb-8">
      <h2 className="text-sm font-semibold text-foreground mb-3">Recently Visited</h2>
      <div className="flex gap-3 overflow-x-auto pb-1">
        {entries.map((entry) => (
          <Link
            key={entry.ticker}
            href={`/stock/${encodeURIComponent(entry.ticker)}`}
            className="shrink-0 rounded-lg border border-border bg-surface px-4 py-2.5 hover:border-accent/50 hover:bg-surface-2 transition-colors"
          >
            <div className="text-sm font-medium text-foreground whitespace-nowrap">
              {entry.name}
            </div>
            <div className="text-xs text-muted font-mono">{entry.ticker}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}
