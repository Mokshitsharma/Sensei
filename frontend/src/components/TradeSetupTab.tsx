import type { TradeSetup } from "@/lib/types";
import { Card } from "./ui/Card";
import { Tabs } from "./ui/Tabs";
import { TradeSetupCard } from "./TradeSetupCard";

export function TradeSetupTab({
  intraday,
  swing,
}: {
  intraday: TradeSetup;
  swing: TradeSetup;
}) {
  return (
    <Card title="Trade Setup">
      <Tabs
        defaultIndex={1}
        tabs={[
          { label: "Intraday (15-min)", content: <TradeSetupCard setup={intraday} /> },
          { label: "Swing (Daily)", content: <TradeSetupCard setup={swing} /> },
        ]}
      />
    </Card>
  );
}
