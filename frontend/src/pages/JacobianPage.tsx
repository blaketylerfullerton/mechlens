import { useJacobian } from "@/hooks/useJacobian";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export function JacobianPage() {
  const {
    prompt,
    setPrompt,
    targetToken,
    setTargetToken,
    topK,
    setTopK,
    result,
    error,
    loading,
    runJacobian,
  } = useJacobian();

  return (
    <>
      <div className="border-b border-border bg-muted/50">
        <div className="mx-auto max-w-[100rem] px-8 py-4">
          <h1 className="text-sm font-medium text-foreground">Jacobian lens</h1>
          <p className="text-sm text-muted-foreground">
            Per layer, the gradient of a target token's logit w.r.t. that layer's
            hidden state, decoded through the unembedding — which vocab
            directions are causally pushing toward the prediction.
          </p>
        </div>
      </div>
      <main className="mx-auto grid max-w-[100rem] grid-cols-1 gap-6 p-8 lg:grid-cols-2 lg:items-start">
        <Card>
          <CardHeader>
            <CardTitle>Run the Jacobian lens</CardTitle>
            <CardDescription>
              Leave target token blank to use the model's own top prediction.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="prompt">Prompt</Label>
              <Textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter a prompt…"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="target-token">Target token (optional)</Label>
              <Input
                id="target-token"
                value={targetToken}
                onChange={(e) => setTargetToken(e.target.value)}
                placeholder="e.g. ' Paris' — must tokenize to a single token"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="top-k">Top-k aligned tokens</Label>
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
            <Button onClick={runJacobian} disabled={loading || !prompt.trim()}>
              {loading ? "Running…" : "Run"}
            </Button>
          </CardFooter>
        </Card>

        <div className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Jacobian lens failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading && (
            <Card>
              <CardHeader>
                <CardTitle>Running…</CardTitle>
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
                <CardTitle>Result</CardTitle>
                <CardDescription>
                  Target token:{" "}
                  <Badge className="ml-1 font-mono">
                    {result.target_token} ({result.target_token_id})
                  </Badge>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="max-h-[32rem] overflow-auto rounded-lg border border-border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-20">Layer</TableHead>
                        <TableHead className="w-28">Grad norm</TableHead>
                        <TableHead>Top aligned tokens</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.layers.map((layer, i) => (
                        <TableRow key={i}>
                          <TableCell className="align-top font-mono text-muted-foreground">
                            {i}
                          </TableCell>
                          <TableCell className="align-top font-mono text-muted-foreground">
                            {layer.grad_norm.toFixed(3)}
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1.5">
                              {layer.top_aligned_tokens.map((entry, j) => (
                                <Badge
                                  key={j}
                                  variant={j === 0 ? "default" : "outline"}
                                  className="font-mono"
                                >
                                  {entry.token} · {entry.score.toFixed(2)}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </>
  );
}
