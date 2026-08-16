"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Quote } from "@/lib/types";
import { Change } from "./ui/Badge";

type Candidate = { name: string; ticker: string };

const CANDIDATES: Candidate[] = [
  { name: "SBI", ticker: "SBIN.NS" },
  { name: "Reliance", ticker: "RELIANCE.NS" },
  { name: "TCS", ticker: "TCS.NS" },
  { name: "HDFC Bank", ticker: "HDFCBANK.NS" },
  { name: "Infosys", ticker: "INFY.NS" },
  { name: "Tata Motors", ticker: "TATAMOTORS.NS" },
  { name: "ICICI Bank", ticker: "ICICIBANK.NS" },
  { name: "Adani Enterprises", ticker: "ADANIENT.NS" },
];

const ROTATE_MS = 4500;
const FADE_MS = 300;

export function FeaturedStockTicker({
  initial,
}: {
  initial?: Candidate & { quote: Quote };
}) {
  const list = useMemo(
    () =>
      initial
        ? [initial, ...CANDIDATES.filter((c) => c.ticker !== initial.ticker)]
        : CANDIDATES,
    [initial]
  );

  const cache = useRef<Map<string, Quote>>(new Map());
  if (initial?.quote) cache.current.set(initial.ticker, initial.quote);

  const [index, setIndex] = useState(0);
  const [quote, setQuote] = useState<Quote | null>(initial?.quote ?? null);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const fetchAndCache = async (ticker: string) => {
      if (cache.current.has(ticker)) return cache.current.get(ticker)!;
      try {
        const q = await api.quote(ticker);
        cache.current.set(ticker, q);
        return q;
      } catch {
        return null;
      }
    };

    const current = list[index];
    if (!cache.current.has(current.ticker)) {
      fetchAndCache(current.ticker).then((q) => {
        if (!cancelled) setQuote(q);
      });
    }

    // Fetch the upcoming stock's quote now, in the background, so it's
    // already cached by the time the rotation needs it — the swap can
    // then be an instant fade with no network gap in between.
    const nextIndex = (index + 1) % list.length;
    fetchAndCache(list[nextIndex].ticker);

    const fadeOutTimer = setTimeout(() => {
      if (!cancelled) setVisible(false);
    }, ROTATE_MS - FADE_MS);

    const swapTimer = setTimeout(() => {
      if (cancelled) return;
      setQuote(cache.current.get(list[nextIndex].ticker) ?? null);
      setIndex(nextIndex);
      setVisible(true);
    }, ROTATE_MS);

    return () => {
      cancelled = true;
      clearTimeout(fadeOutTimer);
      clearTimeout(swapTimer);
    };
  }, [index, list]);

  const current = list[index];
  if (!quote) return null;

  return (
    <Link
      href={`/stock/${encodeURIComponent(current.ticker)}`}
      className={`mt-8 inline-flex items-center gap-3 rounded-xl border border-border bg-surface px-5 py-3 transition-all duration-300 hover:border-accent/50 hover:bg-surface-2 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-1"
      }`}
    >
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-green" />
      </span>
      <span className="text-sm font-semibold text-foreground">{current.name}</span>
      <span className="font-mono text-sm font-semibold text-foreground">
        ₹{quote.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </span>
      <Change value={quote.change} pct={quote.pct} showValue={false} />
      <span className="text-xs text-muted">See full AI analysis →</span>
    </Link>
  );
}
