import Link from "next/link";
import type { StockListItem } from "@/lib/types";
import { SearchBar } from "./SearchBar";

export function TopBar({ stocks }: { stocks: StockListItem[] }) {
  return (
    <header className="flex items-center gap-3 border-b border-border bg-surface px-3 sm:px-4 py-3">
      <Link href="/" className="text-lg sm:text-xl font-bold text-accent shrink-0">
        Sensei AI
      </Link>
      <div className="flex-1 min-w-0">
        <SearchBar stocks={stocks} />
      </div>
      <div className="flex items-center gap-3 sm:gap-4 shrink-0 text-muted">
        <button aria-label="Notifications" className="hover:text-foreground">
          🔔
        </button>
        <button aria-label="Settings" className="hover:text-foreground">
          ⚙
        </button>
        <div className="h-8 w-8 rounded-full bg-accent/20 border border-accent/40 flex items-center justify-center text-xs text-accent font-semibold">
          U
        </div>
      </div>
    </header>
  );
}
