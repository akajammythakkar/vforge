export interface Project {
  id: string;
  name: string;
  description: string | null;
  domain: string | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetRow {
  id: string;
  instruction: string;
  input: string | null;
  output: string;
  system_prompt: string | null;
  quality_score: number | null;
  position: number;
  created_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  format: "alpaca" | "sharegpt" | "jsonl";
  source: "generated" | "uploaded" | "hybrid";
  row_count: number;
  created_at: string;
  updated_at: string;
}

export interface TrainingRun {
  id: string;
  project_id: string;
  dataset_id: string | null;
  name: string;
  base_model: string;
  hardware: "tpu" | "gpu";
  method: "lora" | "qlora" | "full";
  status: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  notebook_path: string | null;
  artifact_uri: string | null;
  created_at: string;
  updated_at: string;
}

export interface BenchmarkResult {
  id: string;
  training_run_id: string | null;
  name: string;
  hardware: string;
  benchmark_type: "training" | "inference";
  model: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  notes: string | null;
  created_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}
