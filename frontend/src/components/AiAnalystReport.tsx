import type { Decision } from "@/lib/types";
import { Card } from "./ui/Card";
import { Badge } from "./ui/Badge";
import { Tabs } from "./ui/Tabs";
import { ShapChart } from "./ShapChart";
import { LockedSection } from "./ui/LockedOverlay";

const ACTION_TONE = {
  BUY: "green",
  SELL: "red",
  HOLD: "amber",
} as const;

export function AiAnalystReport({
  decision,
  isSignedIn,
}: {
  decision: Decision;
  isSignedIn: boolean;
}) {
  const narrative = decision.narrative;

  if (!narrative) {
    return (
      <Card title="AI Analyst Report">
        <p className="text-sm text-muted">
          {decision.explanation ?? "No explanation available."}
        </p>
      </Card>
    );
  }

  return (
    <Card
      title="✨ AI Analyst Report"
      badge={
        isSignedIn ? (
          <Badge tone={ACTION_TONE[decision.action]}>
            {decision.action === "BUY"
              ? "STRONG BUY"
              : decision.action === "SELL"
                ? "STRONG SELL"
                : "HOLD"}
          </Badge>
        ) : (
          <Badge>🔒 Locked</Badge>
        )
      }
    >
      <LockedSection
        locked={!isSignedIn}
        title="Sign in to read the full report"
        description="Trend, momentum, volatility, and SHAP feature analysis — free with Google."
        minHeight={360}
      >
        <p className="text-sm font-medium text-foreground mb-5">
          Verdict: {narrative.headline}
        </p>

        <Tabs
          tabs={[
            {
              label: "Trend",
              content: <Prose text={narrative.trend} />,
            },
            {
              label: "Momentum",
              content: <Prose text={narrative.momentum} />,
            },
            {
              label: "Volatility",
              content: <Prose text={narrative.volatility} />,
            },
            {
              label: "AI Models",
              content: <Prose text={narrative.ai_models} />,
            },
            {
              label: "SHAP Drivers",
              content: (
                <div className="space-y-4">
                  <ShapChart items={decision.shap_ranked ?? []} />
                  <Prose text={narrative.shap_story} />
                </div>
              ),
            },
            {
              label: "News",
              content: <Prose text={narrative.news} />,
            },
          ]}
        />

        {narrative.summary && (
          <div className="mt-6 rounded-lg border border-border bg-surface-2 p-4">
            <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-1">
              Summary
            </p>
            <p className="text-sm text-foreground">{narrative.summary}</p>
          </div>
        )}
      </LockedSection>
    </Card>
  );
}

function Prose({ text }: { text?: string }) {
  return <p className="text-sm leading-relaxed text-foreground/90">{text ?? "—"}</p>;
}
