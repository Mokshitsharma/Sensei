import type { NewsResponse, NewsPriceForecast } from "@/lib/types";
import { Card } from "./ui/Card";
import { Badge } from "./ui/Badge";
import { StatGrid, StatTile } from "./ui/StatTile";

const IMPACT_ICON: Record<string, string> = {
  Earnings: "💰",
  Regulatory: "⚖️",
  Management: "👔",
  Macro: "🌍",
  Sector: "🏭",
  General: "📄",
};

const LABEL_TONE = {
  POSITIVE: "green",
  NEGATIVE: "red",
  NEUTRAL: "amber",
} as const;

export function NewsPanel({
  news,
  forecast,
}: {
  news: NewsResponse;
  forecast: NewsPriceForecast;
}) {
  const total = news.bull_count + news.bear_count + news.neutral_count;

  if (total === 0) {
    return (
      <Card title="News Intelligence">
        <p className="text-sm text-muted">No news found for this stock.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-5">
      <Card title="News Intelligence">
        <StatGrid>
          <StatTile
            label="Weighted Score"
            value={`${news.weighted_score >= 0 ? "+" : ""}${news.weighted_score.toFixed(3)}`}
            color={news.weighted_score > 0.1 ? "green" : news.weighted_score < -0.1 ? "red" : "amber"}
          />
          <StatTile label="Bullish" value={String(news.bull_count)} color="green" />
          <StatTile label="Bearish" value={String(news.bear_count)} color="red" />
          <StatTile label="Neutral" value={String(news.neutral_count)} />
        </StatGrid>

        <p className="text-sm text-foreground/90 mt-4">{news.summary}</p>

        <div className="grid md:grid-cols-2 gap-4 mt-5">
          {news.top_bullish && (
            <HeadlineHighlight label="Top Bullish" tone="green" item={news.top_bullish} />
          )}
          {news.top_bearish && (
            <HeadlineHighlight label="Top Bearish" tone="red" item={news.top_bearish} />
          )}
        </div>

        <details className="mt-5">
          <summary className="cursor-pointer text-sm font-medium text-accent">
            All {total} headlines
          </summary>
          <div className="mt-3 divide-y divide-border">
            {news.details.map((item, i) => (
              <div key={i} className="py-3">
                <div className="flex items-center gap-2 text-xs mb-1">
                  <Badge tone={LABEL_TONE[item.label]}>{item.label}</Badge>
                  <span className="text-muted">
                    {IMPACT_ICON[item.impact_type ?? "General"] ?? "📄"}{" "}
                    {item.impact_type} &middot; conf {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm">
                  {item.headline}{" "}
                  {item.url && (
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent"
                    >
                      ↗
                    </a>
                  )}
                </p>
                <p className="text-xs text-muted mt-1">
                  {item.source} &middot; {item.published}
                </p>
              </div>
            ))}
          </div>
        </details>
      </Card>

      <Card title="News-driven Price Forecast">
        <StatGrid>
          <StatTile
            label="Direction"
            value={forecast.direction}
            color={
              forecast.direction === "UP"
                ? "green"
                : forecast.direction === "DOWN"
                  ? "red"
                  : "amber"
            }
          />
          <StatTile label="Predicted Price" value={`₹${forecast.predicted_price.toLocaleString()}`} />
          <StatTile
            label="Range"
            value={`₹${forecast.price_low.toLocaleString()}–${forecast.price_high.toLocaleString()}`}
          />
          <StatTile
            label="Expected Move"
            value={`${forecast.expected_move_pct >= 0 ? "+" : ""}${forecast.expected_move_pct.toFixed(2)}%`}
          />
        </StatGrid>
        <p className="text-xs text-muted mt-3">
          Confidence: {forecast.confidence} &middot; Horizon: {forecast.horizon_label}
        </p>
        <p className="text-sm text-foreground/90 mt-2">{forecast.explanation}</p>
      </Card>
    </div>
  );
}

function HeadlineHighlight({
  label,
  tone,
  item,
}: {
  label: string;
  tone: "green" | "red";
  item: NonNullable<NewsResponse["top_bullish"]>;
}) {
  return (
    <div
      className={`rounded-lg border-l-4 p-3 ${
        tone === "green" ? "border-green bg-green/5" : "border-red bg-red/5"
      }`}
    >
      <p className={`text-xs font-semibold mb-1 ${tone === "green" ? "text-green" : "text-red"}`}>
        {label.toUpperCase()} — {item.impact_type}
      </p>
      <p className="text-sm">{item.headline}</p>
      <p className="text-xs text-muted mt-1">
        {item.source} &middot; {item.published}
      </p>
    </div>
  );
}
