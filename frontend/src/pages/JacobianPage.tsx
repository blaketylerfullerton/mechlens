import { useEffect, useRef, useState } from "react";
import { useJacobian } from "@/hooks/useJacobian";
import { JacobianTranscript } from "@/components/jacobian-transcript";
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
    messages,
    draft,
    setDraft,
    targetToken,
    setTargetToken,
    topK,
    setTopK,
    result,
    error,
    loading,
    sendMessage,
    reset,
  } = useJacobian();

  // null = "not touched yet" -> defaults to the last position (the token
  // the target logit is actually read at), clamped once a result with a
  // shorter sequence comes back.
  const [selectedPosition, setSelectedPosition] = useState<number | null>(
    null,
  );
  const positionCount = result?.tokens.length ?? 0;
  const position =
    positionCount === 0
      ? 0
      : Math.min(selectedPosition ?? positionCount - 1, positionCount - 1);

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, loading]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  }

  return (
    <>
      <div className="border-b border-border bg-muted/50">
        <div className="mx-auto max-w-[100rem] px-8 py-4">
          <h1 className="text-sm font-medium text-foreground">Jacobian lens</h1>
          <p className="text-sm text-muted-foreground">
            Per layer and sequence position, the gradient of a target token's
            logit w.r.t. that layer's hidden state, decoded through the
            unembedding — which vocab directions are causally pushing toward
            the prediction.
          </p>
        </div>
      </div>
      <main className="mx-auto grid max-w-[100rem] grid-cols-1 gap-6 p-8 lg:grid-cols-2 lg:items-start">
        <Card className="flex h-[calc(100vh-10rem)] flex-col">
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle>Chat</CardTitle>
                <CardDescription>
                  Each message you send appends to the conversation and
                  re-runs the lens on the full transcript so far.
                </CardDescription>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={reset}
                disabled={messages.length === 0 && !draft}
              >
                Reset
              </Button>
            </div>
            <div className="flex flex-wrap items-end gap-3 pt-2">
              <div className="space-y-1.5">
                <Label htmlFor="target-token" className="text-xs">
                  Target token (optional)
                </Label>
                <Input
                  id="target-token"
                  value={targetToken}
                  onChange={(e) => setTargetToken(e.target.value)}
                  placeholder="e.g. ' Paris'"
                  className="h-8 w-48"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="top-k" className="text-xs">
                  Top-k
                </Label>
                <Input
                  id="top-k"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="h-8 w-20"
                />
              </div>
            </div>
          </CardHeader>

          <CardContent className="min-h-0 flex-1 overflow-hidden">
            <div
              ref={scrollRef}
              className="h-full space-y-3 overflow-y-auto"
            >
              {messages.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  Send a message to start the conversation — the lens re-runs
                  on the whole transcript after every turn. Hover a word once
                  a result comes back to see its next-token prediction.
                </p>
              )}
              <JacobianTranscript messages={messages} result={result} />
              {loading && (
                <div className="flex justify-start">
                  <div className="w-2/3 space-y-2 rounded-lg bg-muted px-3 py-2">
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3 w-2/3" />
                  </div>
                </div>
              )}
            </div>
          </CardContent>

          <CardFooter className="gap-2 border-t border-border pt-4">
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
              className="min-h-[2.5rem] resize-none"
              rows={1}
            />
            <Button onClick={() => void sendMessage()} disabled={loading || !draft.trim()}>
              {loading ? "Thinking…" : "Send"}
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

          {!result && !error && (
            <Card>
              <CardHeader>
                <CardTitle>No result yet</CardTitle>
                <CardDescription>
                  Send a message on the left to run the Jacobian lens.
                </CardDescription>
              </CardHeader>
            </Card>
          )}

          {result && (
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
              <CardContent className="space-y-3">
                <div className="space-y-1.5">
                  <Label>Position</Label>
                  <div className="flex flex-wrap gap-1">
                    {result.tokens.map((token, i) => (
                      <Button
                        key={i}
                        type="button"
                        size="sm"
                        variant={i === position ? "default" : "outline"}
                        className="h-7 px-2 font-mono"
                        onClick={() => setSelectedPosition(i)}
                      >
                        {token}
                      </Button>
                    ))}
                  </div>
                </div>
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
                      {result.layers.map((layer, i) => {
                        const atPosition = layer.positions[position];
                        return (
                          <TableRow key={i}>
                            <TableCell className="align-top font-mono text-muted-foreground">
                              {i}
                            </TableCell>
                            <TableCell className="align-top font-mono text-muted-foreground">
                              {atPosition.grad_norm.toFixed(3)}
                            </TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1.5">
                                {atPosition.top_aligned_tokens.map(
                                  (entry, j) => (
                                    <Badge
                                      key={j}
                                      variant={j === 0 ? "default" : "outline"}
                                      className="font-mono"
                                    >
                                      {entry.token} · {entry.score.toFixed(2)}
                                    </Badge>
                                  ),
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        );
                      })}
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
