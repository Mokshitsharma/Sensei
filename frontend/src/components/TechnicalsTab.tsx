import type { Decision, Signals, SupportResistance as SrType } from "@/lib/types";
import { AiAnalystReport } from "./AiAnalystReport";
import { RawModelSignals } from "./RawModelSignals";
import { SupportResistance } from "./SupportResistance";

export function TechnicalsTab({
  signals,
  decision,
  supportResistance,
}: {
  signals: Signals;
  decision: Decision;
  supportResistance: SrType;
}) {
  return (
    <div className="space-y-5">
      <div className="grid lg:grid-cols-3 gap-5 items-start">
        <div className="lg:col-span-2">
          <AiAnalystReport decision={decision} />
        </div>
        <RawModelSignals signals={signals} decision={decision} />
      </div>
      <SupportResistance data={supportResistance} />
    </div>
  );
}
