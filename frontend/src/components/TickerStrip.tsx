import type { Indices } from "@/lib/types";
import { Change } from "./ui/Badge";

export function TickerStrip({ indices }: { indices: Indices }) {
  const entries = Object.entries(indices);
  if (entries.length === 0) return null;

  return (
    <div className="flex items-center gap-6 overflow-x-auto border-b border-border bg-surface px-4 py-2 text-sm">
      {entries.map(([name, quote]) => (
        <div key={name} className="flex items-center gap-2 whitespace-nowrap">
          <span className="text-muted font-medium">{name}</span>
          {quote ? (
            <>
              <span className="font-mono font-semibold">
                {quote.value.toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })}
              </span>
              <Change value={quote.change} pct={quote.pct} />
            </>
          ) : (
            <span className="text-muted">—</span>
          )}
        </div>
      ))}
    </div>
  );
}
