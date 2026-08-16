import type { TradeSetup } from "@/lib/types";
import { Card } from "./ui/Card";
import { Tabs } from "./ui/Tabs";
import { TradeSetupCard } from "./TradeSetupCard";
import { LockedSection } from "./ui/LockedOverlay";

export function TradeSetupTab({
  intraday,
  swing,
  isSignedIn,
}: {
  intraday: TradeSetup;
  swing: TradeSetup;
  isSignedIn: boolean;
}) {
  return (
    <Card title="Trade Setup">
      <LockedSection
        locked={!isSignedIn}
        title="Sign in to see trade setups"
        description="Intraday and swing entry zones, stop-loss, and targets — free with Google."
        minHeight={420}
      >
        <Tabs
          defaultIndex={1}
          tabs={[
            { label: "Intraday (15-min)", content: <TradeSetupCard setup={intraday} /> },
            { label: "Swing (Daily)", content: <TradeSetupCard setup={swing} /> },
          ]}
        />
      </LockedSection>
    </Card>
  );
}
