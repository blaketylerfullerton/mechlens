import { useState } from "react";
import type { TraceResult } from "@/types";

export function useTrace(initialPrompt = "The capital of France is") {
  const [prompt, setPrompt] = useState(initialPrompt);
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

  return { prompt, setPrompt, topK, setTopK, result, error, loading, runTrace };
}
