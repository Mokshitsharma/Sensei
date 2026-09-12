"use client";

import { useState } from "react";
import { useAuth, SignInButton } from "@clerk/nextjs";
import { api } from "@/lib/api";

export function PaperTradePanel({
  ticker,
  currentPrice,
  suggestedEntry,
}: {
  ticker: string;
  currentPrice: number;
  suggestedEntry?: number;
}) {
  const { isSignedIn, getToken } = useAuth();
  const [quantity, setQuantity] = useState(1);
  const [status, setStatus] = useState<
    | { kind: "idle" }
    | { kind: "submitting" }
    | { kind: "success"; side: "BUY" | "SELL" }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  async function submit(side: "BUY" | "SELL") {
    setStatus({ kind: "submitting" });
    try {
      const token = await getToken();
      if (!token) throw new Error("Not signed in");
      await api.trade(token, { ticker, side, quantity });
      setStatus({ kind: "success", side });
    } catch (err) {
      setStatus({
        kind: "error",
        message: err instanceof Error ? err.message : "Trade failed",
      });
    }
  }

  if (!isSignedIn) {
    return (
      <div className="rounded-xl border border-border bg-surface p-5">
        <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
          Paper Trade
        </p>
        <p className="text-sm text-muted mb-4">
          Sign in to test this call with a simulated ₹1,00,000 portfolio —
          no real money.
        </p>
        <SignInButton mode="modal">
          <button className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover">
            Sign in free
          </button>
        </SignInButton>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
        Paper Trade
      </p>

      <div className="flex items-center gap-2 mb-3">
        <label className="text-xs text-muted">Qty</label>
        <input
          type="number"
          min={1}
          value={quantity}
          onChange={(e) => setQuantity(Math.max(1, Number(e.target.value) || 1))}
          className="w-20 rounded-md border border-border bg-surface-2 px-2 py-1.5 text-sm font-mono text-foreground"
        />
        <span className="text-xs text-muted font-mono ml-auto">
          @ ₹{currentPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </span>
      </div>

      {suggestedEntry !== undefined && (
        <p className="mb-3 text-xs text-muted">
          AI&apos;s suggested entry:{" "}
          <span className="text-accent font-mono">
            ₹{suggestedEntry.toFixed(2)}
          </span>{" "}
          — Buy/Sell below fill at the current live price.
        </p>
      )}

      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => submit("BUY")}
          disabled={status.kind === "submitting"}
          className="rounded-lg bg-green/10 border border-green/30 text-green px-4 py-2 text-sm font-semibold hover:bg-green/20 disabled:opacity-50"
        >
          Buy
        </button>
        <button
          onClick={() => submit("SELL")}
          disabled={status.kind === "submitting"}
          className="rounded-lg bg-red/10 border border-red/30 text-red px-4 py-2 text-sm font-semibold hover:bg-red/20 disabled:opacity-50"
        >
          Sell
        </button>
      </div>

      {status.kind === "success" && (
        <p className="mt-3 text-xs text-green">
          {status.side === "BUY" ? "Bought" : "Sold"} {quantity} share
          {quantity > 1 ? "s" : ""}. View in{" "}
          <a href="/portfolio" className="underline">
            Portfolio
          </a>
          .
        </p>
      )}
      {status.kind === "error" && (
        <p className="mt-3 text-xs text-red">{status.message}</p>
      )}

      <p className="mt-3 text-[11px] text-muted leading-relaxed">
        Simulated fill at the current live price. No broker, no real money.
      </p>
    </div>
  );
}
