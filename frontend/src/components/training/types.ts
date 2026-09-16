export type Options = { d_in?: number; multi_layer?: boolean; storage_management?: boolean; model_downloads?: boolean; model: string; model_repository: string | null; layers: number | null; device: string; readiness: string; estimate_note: string; supported_models: string[] }
export type Check = { status: string; reason?: string; definition?: string; max_loss_delta?: number }
export type Evaluation = { [key: string]: unknown; identity_substitution?: Check }
export type Run = {
  batch_id?: string; batch_index?: number; batch_size?: number; storage_warning?: string;
  id: string; status: string; phase: string; created_at: number; tokens: number; error: string | null;
  config: { layer: number; features: number; training_tokens: number; dataset: string; compare_huggingface?: boolean };
  checkpoint: { artifact_id: string; tokens: number } | null;
  evaluation: Evaluation | null;
  validation?: { model_agreement?: Check; checkpoint_agreement?: Check; activation_identity?: Check; identity_substitution?: Check };
  validation_progress?: { completed: number; total: number } | null;
  resume_supported: boolean;
  examples?: { sequences_scanned: number; tokens_scanned: number } | null;
}
export type InterpJob = { id: string; run_id: string; artifact_id: string; feature_ids: number[]; endpoint: string; model: string; status: string; phase: string; completed: number; total: number; error: string | null; created_at: number }
export type ExampleReport = { artifact_id: string; sequences_scanned: number; tokens_scanned: number; examples: Record<string, { activation: number; context: string; token_position: number }[]> }
export type Metric = { seq: number; tokens: number; loss: number; mse: number; explained_variance: number;
  l0: number; dead_fraction: number; tokens_per_second: number; eta_s: number; elapsed_s: number }
export type StorageRecord = { id: string; model: string | null; layer: number; features: number; status: string;
  created_at: number; has_checkpoint: boolean; bytes: number; delete_blocked_reason: string | null }
export type Storage = { runs: StorageRecord[]; total_bytes: number; free_bytes: number; checkpoint_policy: string }
export type ModelDownload = { repository: string; name: string; bytes: number | null; downloaded: boolean; loaded: boolean }
export type ModelDownloads = { models: ModelDownload[]; total_bytes: number; scan_error: string | null }
