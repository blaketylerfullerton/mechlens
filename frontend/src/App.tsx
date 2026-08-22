import { useState } from "react";
import type { TraceResult } from "./types";
import { Button } from "@/components/ui/button";

export default function App() {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [result, setResult] = useState<TraceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runTrace() {
    setError(null);
    try {
      // Placeholder endpoint — wire this up once backend/app/api exposes /trace.
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <h1 className="text-2xl font-semibold mb-4">mechlens</h1>
      <div className="flex gap-2 mb-6">
        <input
          className="flex-1 rounded bg-slate-800 px-3 py-2 outline-none focus:ring-2 focus:ring-teal-500"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <Button onClick={runTrace}>Trace</Button>
      </div>
      {error && <p className="text-red-400">{error}</p>}
      {result && (
        <pre className="text-xs bg-slate-900 rounded p-4 overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
