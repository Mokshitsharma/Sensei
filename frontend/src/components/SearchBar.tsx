"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { StockListItem } from "@/lib/types";

export function SearchBar({ stocks }: { stocks: StockListItem[] }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    return stocks
      .filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          s.ticker.toLowerCase().includes(q)
      )
      .slice(0, 8);
  }, [query, stocks]);

  function select(ticker: string) {
    setOpen(false);
    setQuery("");
    router.push(`/stock/${encodeURIComponent(ticker)}`);
  }

  return (
    <div className="relative w-full max-w-sm" ref={containerRef}>
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-3 py-2">
        <span className="text-muted text-sm">🔍</span>
        <input
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search a stock…"
          className="w-full bg-transparent text-sm outline-none placeholder:text-muted"
        />
      </div>
      {open && results.length > 0 && (
        <div className="absolute z-20 mt-1 w-full rounded-lg border border-border bg-surface shadow-xl overflow-hidden">
          {results.map((s) => (
            <button
              key={s.ticker}
              onMouseDown={() => select(s.ticker)}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-surface-2"
            >
              <span>{s.name}</span>
              <span className="text-muted font-mono text-xs">{s.ticker}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
