import type { ChatMessage } from "@/hooks/useJacobian";
import type { JacobianLensResult } from "@/types";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";

interface Segment {
  role: "user" | "assistant";
  indices: number[];
}

// Chat-templated prompts (the normal case, once the backend has a real
// chat_template) look like:
//   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...
// Walking those markers recovers exactly which token indices belong to
// which turn, in the same order `messages` was sent in — one segment per
// message, system turns dropped. Returns null (not just []) when no
// <|im_start|> marker shows up at all, signaling "this prompt wasn't
// templated" so the caller falls back to plain rendering instead of
// mis-parsing raw text as one giant segment.
function parseSegments(tokens: string[]): Segment[] | null {
  const segments: Segment[] = [];
  let sawTemplateMarker = false;
  let i = 0;

  while (i < tokens.length) {
    if (tokens[i] !== "<|im_start|>") {
      i += 1;
      continue;
    }
    sawTemplateMarker = true;
    i += 1;

    // The role name isn't reliably one token — "assistant" tokenizes as
    // "ass" + "istant" — so accumulate pieces until the template's own
    // newline separator instead of assuming a fixed offset.
    let role = "";
    while (i < tokens.length && tokens[i] !== "\n") {
      role += tokens[i];
      i += 1;
    }
    if (tokens[i] === "\n") i += 1;

    const indices: number[] = [];
    while (i < tokens.length && tokens[i] !== "<|im_end|>") {
      indices.push(i);
      i += 1;
    }
    if (tokens[i] === "<|im_end|>") i += 1;

    if (indices.length > 0 && (role === "user" || role === "assistant")) {
      segments.push({ role, indices });
    }
  }

  return sawTemplateMarker ? segments : null;
}

function isWhitespace(token: string) {
  return token.trim().length === 0;
}

function Bubble({
  role,
  children,
}: {
  role: "user" | "assistant";
  children: React.ReactNode;
}) {
  return (
    <div className={cn("flex", role === "user" ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 font-mono text-sm leading-relaxed",
          role === "user"
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
        )}
      >
        {children}
      </div>
    </div>
  );
}

function TokenSpan({
  token,
  predictions,
}: {
  token: string;
  predictions: JacobianLensResult["next_token_predictions"][number] | undefined;
}) {
  if (isWhitespace(token) || !predictions || predictions.top_predictions.length === 0) {
    return <span>{token}</span>;
  }

  return (
    <HoverCard>
      <HoverCardTrigger
        delay={150}
        closeDelay={0}
        render={<span />}
        className="cursor-pointer rounded-sm underline decoration-dotted decoration-muted-foreground/50 underline-offset-2 hover:bg-foreground/15"
      >
        {token}
      </HoverCardTrigger>
      <HoverCardContent className="w-80" side="top" align="start">
        <p className="mb-2 text-xs text-muted-foreground">
          Next-token prediction after{" "}
          <span className="font-mono text-foreground">"{token.trim()}"</span>
        </p>
        <div className="space-y-2">
          {predictions.top_predictions.map((p, i) => (
            <div key={i} className="space-y-0.5">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="truncate font-mono">
                  {p.token.trim() || "(space)"}
                </span>
                <span className="tabular-nums text-muted-foreground">
                  {(p.prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${Math.max(p.prob * 100, 2)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

export function JacobianTranscript({
  messages,
  result,
}: {
  messages: ChatMessage[];
  result: JacobianLensResult | null;
}) {
  const segments = result ? parseSegments(result.tokens) : null;

  if (!segments) {
    return (
      <>
        {messages.map((message, i) => (
          <Bubble key={i} role={message.role}>
            {message.content}
          </Bubble>
        ))}
      </>
    );
  }

  // `result` always lags one turn behind `messages` (it explains the
  // prompt a reply was generated from, not the reply itself), so whatever
  // messages the parsed segments don't cover yet renders as a plain bubble
  // until the next send folds them into an annotated result.
  const trailing = messages.slice(segments.length);

  return (
    <>
      {segments.map((segment, i) => (
        <Bubble key={i} role={segment.role}>
          {segment.indices.map((idx) => (
            <TokenSpan
              key={idx}
              token={result!.tokens[idx]}
              predictions={result!.next_token_predictions[idx]}
            />
          ))}
        </Bubble>
      ))}
      {trailing.map((message, i) => (
        <Bubble key={`trailing-${i}`} role={message.role}>
          {message.content}
        </Bubble>
      ))}
    </>
  );
}
