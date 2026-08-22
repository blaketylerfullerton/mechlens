// Mirrors TraceResult in backend/app/engine.py — keep in sync by hand
// until the backend has an OpenAPI schema to generate from.

export interface LogitLensEntry {
  token: string;
  prob: number;
}

export interface TraceResult {
  prompt: string;
  tokens: string[];
  predicted_next_token: string;
  hidden_states: number[][][]; // [layer][token][dim]
  attention: number[][][][]; // [layer][head][query][key]
  logit_lens: LogitLensEntry[][]; // [layer][top-k], taken at the last position
}
