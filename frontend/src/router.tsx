import { createRootRoute, createRoute, createRouter, Outlet } from "@tanstack/react-router";
import { TopNav } from "@/components/TopNav";
import { TracePage } from "@/pages/TracePage";
import { AttentionPage } from "@/pages/AttentionPage";

const rootRoute = createRootRoute({
  component: () => (
    <div className="min-h-screen bg-background text-foreground">
      <TopNav />
      <Outlet />
    </div>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: TracePage,
});

const attentionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/attention",
  component: AttentionPage,
});

const routeTree = rootRoute.addChildren([indexRoute, attentionRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
