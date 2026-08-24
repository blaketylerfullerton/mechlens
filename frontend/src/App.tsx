import { useState } from "react";
import type { TraceResult } from "./types";
import { Button } from "@/components/ui/button";
import { TopNav } from "@/components/TopNav";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function App() {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<TraceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runTrace() {
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, top_k: topK }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <TopNav />
      <div className="border-b border-border bg-muted/50">
        <div className="mx-auto max-w-[100rem] px-8 py-4">
          <h1 className="text-sm font-medium text-foreground">Trace</h1>
          <p className="text-sm text-muted-foreground">
            Run a forward pass over a prompt and inspect the model's hidden
            states, attention patterns, and logit lens (its top predicted
            tokens if generation stopped) at every layer.
          </p>
        </div>
      </div>
      <main className="mx-auto grid max-w-[100rem] grid-cols-1 gap-6 p-8 lg:grid-cols-2 lg:items-start">
        <Card>
          <CardHeader>
            <CardTitle>Trace a prompt</CardTitle>
            <CardDescription>
              Run a forward pass and inspect the model's hidden states, attention, and logit lens
              at every layer.
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
            <div className="space-y-2">
              <Label htmlFor="top-k">Top-k (logit lens)</Label>
              <Input
                id="top-k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-24"
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
            <Tabs defaultValue="logit-lens">
              <TabsList>
                <TabsTrigger value="logit-lens">Logit lens</TabsTrigger>
                <TabsTrigger value="raw">Raw JSON</TabsTrigger>
              </TabsList>

              <Card className="mt-3">
                <CardHeader>
                  <CardTitle>Result</CardTitle>
                  <CardDescription>
                    Predicted next token:{" "}
                    <Badge className="ml-1 font-mono">{result.predicted_next_token}</Badge>
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="mb-2 text-sm font-medium text-muted-foreground">Tokens</p>
                    <div className="flex flex-wrap gap-1.5">
                      {result.tokens.map((token, i) => (
                        <Badge key={i} variant="secondary" className="font-mono">
                          {token}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  <TabsContent value="logit-lens">
                    <p className="mb-2 text-sm text-muted-foreground">
                      Top-{topK} predicted tokens if generation stopped at each layer.
                    </p>
                    <div className="max-h-96 overflow-auto rounded-lg border border-border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="w-20">Layer</TableHead>
                            <TableHead>Top predictions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.logit_lens.map((layerPreds, layer) => (
                            <TableRow key={layer}>
                              <TableCell className="align-top font-mono text-muted-foreground">
                                {layer}
                              </TableCell>
                              <TableCell>
                                <div className="flex flex-wrap gap-1.5">
                                  {layerPreds.map((entry, i) => (
                                    <Badge
                                      key={i}
                                      variant={i === 0 ? "default" : "outline"}
                                      className="font-mono"
                                    >
                                      {entry.token} · {(entry.prob * 100).toFixed(1)}%
                                    </Badge>
                                  ))}
                                </div>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </TabsContent>
                  <TabsContent value="raw">
                    <pre className="max-h-96 overflow-auto rounded-lg bg-muted p-4 text-xs">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </TabsContent>
                </CardContent>
              </Card>
            </Tabs>
          )}
        </div>
      </main>
    </div>
  );
}
