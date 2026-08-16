const STEPS = [
  { emoji: "🔍", text: "Scanning price action…" },
  { emoji: "🧠", text: "Running LSTM & TCN predictions…" },
  { emoji: "📊", text: "Mapping support & resistance…" },
  { emoji: "📰", text: "Reading the latest news…" },
  { emoji: "⚖️", text: "Weighing the AI verdict…" },
  { emoji: "🎯", text: "Building your trade setup…" },
];

export default function Loading() {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <div className="h-[57px] border-b border-border bg-surface shrink-0" />
      <div className="h-[41px] border-b border-border bg-surface shrink-0" />

      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16 text-center">
        <div className="relative h-14 w-14">
          <div className="absolute inset-0 rounded-full border-2 border-border" />
          <div className="animate-loader-ring absolute inset-0 rounded-full border-2 border-transparent border-t-accent" />
        </div>

        <div className="relative mt-6 h-6 w-full max-w-sm">
          {STEPS.map((step, i) => (
            <p
              key={step.text}
              className="animate-loader-msg absolute inset-0 flex items-center justify-center gap-2 text-sm text-muted"
              style={{ animationDelay: `${i * 2}s` }}
            >
              <span>{step.emoji}</span>
              {step.text}
            </p>
          ))}
        </div>

        <p className="mt-8 text-xs text-muted/70 max-w-xs">
          First look at a stock runs several AI models fresh — after this,
          it's cached and loads instantly for a few minutes.
        </p>
      </div>
    </div>
  );
}
