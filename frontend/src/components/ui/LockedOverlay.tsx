import { SignInButton } from "@clerk/nextjs";

export function LockedOverlay({
  title = "Sign in to unlock the full AI recommendation",
  description = "Entry zones, stop-loss, targets, and the AI's reasoning are free once you sign in — takes a few seconds with Google.",
}: {
  title?: string;
  description?: string;
}) {
  return (
    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-xl bg-background/80 backdrop-blur-sm px-6 text-center">
      <span className="text-2xl">🔒</span>
      <p className="text-sm font-semibold text-foreground max-w-xs">{title}</p>
      <p className="text-xs text-muted max-w-xs">{description}</p>
      <SignInButton mode="modal">
        <button className="mt-1 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover">
          Sign in free
        </button>
      </SignInButton>
    </div>
  );
}

export function LockedSection({
  locked,
  children,
  title,
  description,
  minHeight = 220,
}: {
  locked: boolean;
  children: React.ReactNode;
  title?: string;
  description?: string;
  minHeight?: number;
}) {
  if (!locked) return <>{children}</>;
  return (
    <div className="relative" style={{ minHeight }}>
      <div className="pointer-events-none select-none blur-sm opacity-60">
        {children}
      </div>
      <LockedOverlay title={title} description={description} />
    </div>
  );
}
