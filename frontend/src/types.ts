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

// Mirrors JacobianLensResult in backend/app/engine.py
export interface JacobianAlignedToken {
  token: string;
  score: number;
}

export interface JacobianPositionResult {
  token: string;
  grad_norm: number;
  top_aligned_tokens: JacobianAlignedToken[];
}

export interface JacobianLayerResult {
  positions: JacobianPositionResult[];
}

export interface NextTokenPrediction {
  token: string;
  prob: number;
}

export interface PositionNextTokenPredictions {
  top_predictions: NextTokenPrediction[];
}

export interface JacobianLensResult {
  prompt: string;
  tokens: string[];
  target_token: string;
  target_token_id: number;
  next_token_predictions: PositionNextTokenPredictions[]; // [position], real forward-pass softmax
  layers: JacobianLayerResult[]; // [layer][position]
}
