import { clerkMiddleware } from "@clerk/nextjs/server";

// No hard route protection: /explore and /stock/* are public (browsing,
// search, confidence scores, charts). The AI recommendation itself is
// gated at the content level (see AiSignalPanel, AiAnalystReport,
// TradeSetupTab) so signed-out visitors see real value before signing in.
export default clerkMiddleware();

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
