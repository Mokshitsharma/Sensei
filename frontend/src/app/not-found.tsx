import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <span className="text-2xl font-bold text-accent">Sensei AI</span>
      <h1 className="text-lg font-semibold">Page not found</h1>
      <p className="max-w-sm text-sm text-muted">
        We couldn&apos;t find that stock or page. It may not be covered by
        the intelligence engine yet.
      </p>
      <Link
        href="/explore"
        className="mt-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover"
      >
        Back to Explore
      </Link>
    </div>
  );
}
