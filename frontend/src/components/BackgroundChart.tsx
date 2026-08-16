type Variant = {
  path: string;
  viewBoxH: number;
  className: string;
  rotate: string;
  dur: string;
};

const VARIANTS: Variant[] = [
  {
    // top-right — steady climb
    path: "M0,210 L40,195 L80,205 L120,160 L160,175 L200,120 L240,140 L280,90 L320,105 L360,55 L400,70 L440,20 L480,35",
    viewBoxH: 230,
    className: "top-0 -right-10 sm:-right-8 w-[220px] sm:w-[340px] opacity-40 sm:opacity-55",
    rotate: "perspective(900px) rotateX(10deg) rotateY(-14deg) rotateZ(2deg)",
    dur: "4.5s",
  },
  {
    // top-left — dip then recovery
    path: "M0,60 L40,90 L80,80 L120,130 L160,110 L200,160 L240,140 L280,100 L320,115 L360,70 L400,85 L440,45 L480,60",
    viewBoxH: 200,
    className: "top-6 -left-14 sm:-left-10 w-[170px] sm:w-[260px] opacity-30 sm:opacity-40",
    rotate: "perspective(900px) rotateX(8deg) rotateY(12deg) rotateZ(-2deg)",
    dur: "5.2s",
  },
  {
    // bottom-left — choppy climb
    path: "M0,150 L30,140 L60,155 L90,120 L120,135 L150,95 L180,110 L210,70 L240,90 L270,50 L300,65 L330,30 L360,45",
    viewBoxH: 170,
    className: "bottom-0 -left-10 sm:-left-6 w-[150px] sm:w-[230px] opacity-25 sm:opacity-35",
    rotate: "perspective(900px) rotateX(-8deg) rotateY(-10deg) rotateZ(3deg)",
    dur: "5.8s",
  },
  {
    // bottom-right — gentle climb
    path: "M0,120 L35,110 L70,115 L105,90 L140,98 L175,70 L210,80 L245,55 L280,62 L315,35 L350,42 L385,18 L420,25",
    viewBoxH: 140,
    className: "-bottom-4 right-4 sm:right-10 w-[160px] sm:w-[240px] opacity-25 sm:opacity-35",
    rotate: "perspective(900px) rotateX(-6deg) rotateY(14deg) rotateZ(-1deg)",
    dur: "6.4s",
  },
];

function MiniChart({ v, id }: { v: Variant; id: number }) {
  const w = Number(v.path.match(/L(\d+),/g)?.pop()?.replace(/[L,]/g, "")) || 480;
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute ${v.className}`}
      style={{ transform: v.rotate }}
    >
      <svg
        viewBox={`0 0 ${w} ${v.viewBoxH}`}
        fill="none"
        className="w-full h-auto drop-shadow-[0_16px_32px_rgba(0,0,0,0.15)]"
      >
        <defs>
          <linearGradient id={`bg-chart-fill-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>

        <path
          d={`${v.path} L${w},${v.viewBoxH} L0,${v.viewBoxH} Z`}
          fill={`url(#bg-chart-fill-${id})`}
        />

        <path
          d={v.path}
          stroke="var(--accent)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="animate-bg-chart-draw"
          pathLength={1}
        />

        <circle r="4" fill="var(--accent)" className="animate-bg-chart-dot">
          <animateMotion dur={v.dur} repeatCount="indefinite" path={v.path} rotate="auto" />
        </circle>
      </svg>
    </div>
  );
}

export function BackgroundChart() {
  return (
    <>
      {VARIANTS.map((v, i) => (
        <MiniChart key={i} v={v} id={i} />
      ))}
    </>
  );
}
