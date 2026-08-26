import { useState } from "react";
import type { JacobianLensResult } from "@/types";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const REPLY_MAX_NEW_TOKENS = 60;

export function useJacobian() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [targetToken, setTargetToken] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<JacobianLensResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // The backend applies the model's own chat template (falling back to a
  // plain transcript for models with none) and hands back the exact prompt
  // it generated from, so formatting never has to be duplicated here.
  async function generateReply(
    history: ChatMessage[],
  ): Promise<{ text: string; prompt: string }> {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: history,
        max_new_tokens: REPLY_MAX_NEW_TOKENS,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  // Same templating the reply was generated from, but re-run against the
  // conversation *including* that reply — so the lens (and its per-word
  // hover data) covers the reply too, not just the prompt that led to it.
  async function buildFullPrompt(history: ChatMessage[]): Promise<string> {
    const res = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const { prompt } = await res.json();
    return prompt;
  }

  async function runJacobian(fullPrompt: string) {
    const res = await fetch("/api/jacobian", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: fullPrompt,
        target_token: targetToken.trim() || null,
        top_k: topK,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    setResult(await res.json());
  }

  async function sendMessage() {
    const text = draft.trim();
    if (!text || loading) return;

    const userMessage: ChatMessage = { role: "user", content: text };
    const afterUser = [...messages, userMessage];
    setMessages(afterUser);
    setDraft("");
    setError(null);
    setLoading(true);
    try {
      const { text: reply } = await generateReply(afterUser);
      const afterReply = [...afterUser, { role: "assistant" as const, content: reply }];
      setMessages(afterReply);
      // Analyze the conversation including the reply that was just
      // generated, so hover data covers it immediately instead of only
      // becoming available once the next message is sent.
      const fullPrompt = await buildFullPrompt(afterReply);
      await runJacobian(fullPrompt);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setMessages([]);
    setDraft("");
    setResult(null);
    setError(null);
  }

  return {
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
  };
}
