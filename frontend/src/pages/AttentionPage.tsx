import { useTrace } from "@/hooks/useTrace";
import { AttentionHeatmap } from "@/components/AttentionHeatmap";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

export function AttentionPage() {
  const { prompt, setPrompt, result, error, loading, runTrace } = useTrace();

  return (
    <>
      <div className="border-b border-border bg-muted/50">
        <div className="mx-auto max-w-[100rem] px-8 py-4">
          <h1 className="text-sm font-medium text-foreground">Attention heatmaps</h1>
          <p className="text-sm text-muted-foreground">
            Run a forward pass and visualize per-layer, per-head attention weights between
            tokens.
          </p>
        </div>
      </div>
      <main className="mx-auto grid max-w-[100rem] grid-cols-1 gap-6 p-8 lg:grid-cols-[24rem_1fr] lg:items-start">
        <Card>
          <CardHeader>
            <CardTitle>Trace a prompt</CardTitle>
            <CardDescription>
              The attention weights returned by the trace are visualized as a heatmap on the
              right.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="prompt">Prompt</Label>
              <Textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a prompt to trace…"
              />
            </div>
          </CardContent>
          <CardFooter>
            <Button onClick={runTrace} disabled={loading || !prompt.trim()}>
              {loading ? "Tracing…" : "Trace"}
            </Button>
          </CardFooter>
        </Card>

        <div className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Trace failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading && (
            <Card>
              <CardHeader>
                <CardTitle>Tracing…</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
              </CardContent>
            </Card>
          )}

          {result && !loading && (
            <Card>
              <CardHeader>
                <CardTitle>Attention</CardTitle>
              </CardHeader>
              <CardContent>
                <AttentionHeatmap tokens={result.tokens} attention={result.attention} />
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </>
  );
}
