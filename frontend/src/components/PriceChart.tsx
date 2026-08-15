"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  CandlestickSeries,
  LineSeries,
  type IChartApi,
} from "lightweight-charts";
import type { PriceRecord } from "@/lib/types";

const CHART_HEIGHT = 380;

function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0];
  for (let i = 0; i < values.length; i++) {
    const v = i === 0 ? values[0] : values[i] * k + prev * (1 - k);
    out.push(v);
    prev = v;
  }
  return out;
}

export function PriceChart({ records }: { records: PriceRecord[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || records.length === 0) return;

    // lightweight-charts' `autoSize: true` sometimes never fires its first
    // resize if the container's width isn't stable yet at creation time
    // (e.g. still settling inside a CSS grid) — it's happened here. So
    // size it manually up front and keep it in sync with ResizeObserver.
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8b93a7",
      },
      grid: {
        vertLines: { color: "#232a3b" },
        horzLines: { color: "#232a3b" },
      },
      rightPriceScale: { borderColor: "#232a3b" },
      timeScale: { borderColor: "#232a3b", timeVisible: false },
      width: container.clientWidth,
      height: CHART_HEIGHT,
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#4ade80",
      downColor: "#f87171",
      borderVisible: false,
      wickUpColor: "#4ade80",
      wickDownColor: "#f87171",
    });
    candles.setData(
      records.map((r) => ({
        time: r.date,
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
      }))
    );

    const closes = records.map((r) => r.close);
    const ema20 = ema(closes, 20);
    const ema50 = ema(closes, 50);

    const ema20Series = chart.addSeries(LineSeries, {
      color: "#f97316",
      lineWidth: 1,
    });
    ema20Series.setData(
      records.map((r, i) => ({ time: r.date, value: ema20[i] }))
    );

    const ema50Series = chart.addSeries(LineSeries, {
      color: "#60a5fa",
      lineWidth: 1,
    });
    ema50Series.setData(
      records.map((r, i) => ({ time: r.date, value: ema50[i] }))
    );

    chart.timeScale().fitContent();

    const resizeObserver = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) chart.resize(width, CHART_HEIGHT);
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, [records]);

  return (
    <div
      ref={containerRef}
      className="w-full"
      style={{ height: CHART_HEIGHT }}
    />
  );
}
