import { useState } from "react";
import type { JacobianLensResult } from "@/types";

export function useJacobian(initialPrompt = "The capital of France is") {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [targetToken, setTargetToken] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<JacobianLensResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runJacobian() {
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/jacobian", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          target_token: targetToken.trim() || null,
          top_k: topK,
        }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return {
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
  };
}
