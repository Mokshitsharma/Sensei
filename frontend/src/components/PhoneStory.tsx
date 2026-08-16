function StatusBar() {
  return (
    <div className="flex items-center justify-between px-4 pt-3 text-[10px] font-semibold text-white/70">
      <span>9:41</span>
      <span>●●● 5G 🔋</span>
    </div>
  );
}

function MiniRow({
  name,
  price,
  up,
}: {
  name: string;
  price: string;
  up: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-white/5 px-2.5 py-1.5">
      <span className="text-[10px] font-medium text-white/80">{name}</span>
      <span
        className={`font-mono text-[10px] font-semibold ${up ? "text-[#4ade80]" : "text-[#f87171]"}`}
      >
        {up ? "▲" : "▼"} {price}
      </span>
    </div>
  );
}

function SceneSearchConfused() {
  return (
    <div className="animate-story-scene-a absolute inset-0 flex flex-col px-4 pt-2">
      <StatusBar />
      <div className="mt-4 rounded-lg bg-white/10 px-3 py-2 text-[11px] text-white/60">
        🔍 RELIANCE, TCS, HDFC...
      </div>
      <div className="mt-3 flex flex-col gap-1.5">
        <MiniRow name="RELIANCE" price="2,847" up />
        <MiniRow name="TCS" price="3,912" up={false} />
        <MiniRow name="HDFC BANK" price="1,653" up />
        <MiniRow name="INFY" price="1,489" up={false} />
      </div>
      <div className="mt-6 flex flex-1 flex-col items-center justify-center text-center">
        <span className="text-3xl">😵‍💫</span>
        <p className="mt-2 text-[11px] font-medium text-white/70">
          Too many signals.
          <br />
          Which one do I trust?
        </p>
      </div>
    </div>
  );
}

function SceneAnalyzing() {
  return (
    <div className="animate-story-scene-b absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
      <span className="text-lg font-bold text-[#a3e635]">Sensei AI</span>
      <div className="relative mt-6 h-12 w-12">
        <div className="absolute inset-0 rounded-full border-2 border-white/15" />
        <div className="animate-loader-ring absolute inset-0 rounded-full border-2 border-transparent border-t-[#a3e635]" />
      </div>
      <p className="mt-5 text-[11px] font-medium text-white/70">
        Analyzing RELIANCE.NS…
      </p>
      <p className="mt-1 text-[10px] text-white/40">
        LSTM · TCN · News · SHAP
      </p>
    </div>
  );
}

function ScenePrediction() {
  return (
    <div className="animate-story-scene-c absolute inset-0 flex flex-col px-4 pt-6 text-center">
      <p className="text-[10px] uppercase tracking-wide text-white/40">
        AI Recommendation
      </p>
      <div className="mx-auto mt-2 rounded-md border border-[#4ade80]/40 bg-[#4ade80]/10 px-4 py-1.5">
        <span className="text-sm font-bold text-[#4ade80]">STRONG BUY</span>
      </div>
      <p className="mt-2 text-[11px] text-white/60">87% confidence</p>

      <svg viewBox="0 0 200 70" className="mt-4 w-full" fill="none">
        <path
          d="M0,55 L25,50 L50,52 L75,35 L100,40 L125,20 L150,25 L175,8 L200,12"
          stroke="#4ade80"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <div className="mt-3 flex flex-col gap-1.5 text-left">
        <MiniRow name="Entry" price="2,830–2,855" up />
        <MiniRow name="Target" price="2,960" up />
        <MiniRow name="Stop Loss" price="2,780" up={false} />
      </div>
    </div>
  );
}

function SceneProfit() {
  return (
    <div className="animate-story-scene-d absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
      <span className="text-3xl">🎉</span>
      <p className="mt-3 text-[11px] font-medium text-white/60">
        Target hit
      </p>
      <p className="mt-1 text-2xl font-bold text-[#4ade80] font-mono">
        +₹4,120
      </p>
      <p className="text-[11px] font-mono text-[#4ade80]">+18.4%</p>
      <div className="mt-5 rounded-lg bg-white/5 px-3 py-2 text-[10px] text-white/50">
        ✅ Called by Sensei AI
      </div>
    </div>
  );
}

export function PhoneStory() {
  return (
    <div className="relative mx-auto w-[230px] sm:w-[250px]">
      <div className="relative rounded-[2.5rem] border-[6px] border-[#111] bg-[#111] shadow-2xl shadow-black/30">
        <div className="absolute left-1/2 top-0 z-20 h-4 w-20 -translate-x-1/2 rounded-b-xl bg-[#111]" />
        <div className="relative h-[470px] w-full overflow-hidden rounded-[2rem] bg-gradient-to-b from-[#141414] to-[#0a0a0a]">
          <SceneSearchConfused />
          <SceneAnalyzing />
          <ScenePrediction />
          <SceneProfit />
          <div className="animate-story-overlay pointer-events-none absolute inset-0 z-10 bg-black" />
        </div>
      </div>
      <p className="mt-4 text-center text-xs text-muted">
        Search. Confused. Sensei predicts. You profit.
      </p>
    </div>
  );
}
