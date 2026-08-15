function Pulse({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-surface-2 ${className}`} />;
}

export default function Loading() {
  return (
    <div className="flex min-h-screen w-full">
      <div className="hidden md:block md:w-16 lg:w-56 shrink-0 border-r border-border bg-surface" />
      <div className="flex-1 min-w-0">
        <div className="h-[57px] border-b border-border bg-surface" />
        <div className="h-[41px] border-b border-border bg-surface" />

        <div className="mx-auto max-w-6xl px-6 py-8">
          <Pulse className="h-4 w-28 mb-6" />
          <Pulse className="h-8 w-64 mb-2" />
          <Pulse className="h-6 w-32 mb-6" />
          <Pulse className="h-[380px] w-full mb-6" />

          <div className="grid lg:grid-cols-3 gap-6 items-start">
            <div className="lg:col-span-2 space-y-3">
              <Pulse className="h-9 w-full max-w-md" />
              <Pulse className="h-24 w-full" />
              <Pulse className="h-24 w-full" />
              <Pulse className="h-40 w-full" />
            </div>
            <Pulse className="h-72 w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
