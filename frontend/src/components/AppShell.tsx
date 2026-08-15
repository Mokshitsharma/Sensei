import type { Indices, StockListItem } from "@/lib/types";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { TickerStrip } from "./TickerStrip";

export function AppShell({
  indices,
  stocks,
  children,
}: {
  indices: Indices;
  stocks: StockListItem[];
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen w-full">
      <Sidebar />
      <div className="flex flex-1 flex-col min-w-0">
        <TopBar stocks={stocks} />
        <TickerStrip indices={indices} />
        <main className="flex-1 min-w-0">{children}</main>
      </div>
    </div>
  );
}
