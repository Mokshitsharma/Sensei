"use client";

import Link from "next/link";
import { motion } from "motion/react";
import type { PopularStock } from "@/lib/types";
import { Change } from "./ui/Badge";

export function StockCard({ stock, index = 0 }: { stock: PopularStock; index?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.04, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3 }}
    >
      <Link
        href={`/stock/${encodeURIComponent(stock.ticker)}`}
        className="block rounded-xl border border-border bg-surface p-4 transition-colors hover:border-accent/50 hover:bg-surface-2"
      >
        <div className="text-sm font-semibold text-foreground truncate">
          {stock.name}
        </div>
        <div className="text-xs text-muted font-mono mt-0.5">
          {stock.ticker}
        </div>
        {stock.quote ? (
          <div className="mt-3 flex items-baseline justify-between">
            <span className="font-mono text-base font-semibold">
              ₹{stock.quote.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
            <Change value={stock.quote.change} pct={stock.quote.pct} showValue={false} />
          </div>
        ) : (
          <div className="mt-3 text-xs text-muted">Price unavailable</div>
        )}
      </Link>
    </motion.div>
  );
}
