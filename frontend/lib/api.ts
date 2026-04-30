import type {
  Project,
  Dataset,
  DatasetRow,
  TrainingRun,
  BenchmarkResult,
  ChatMessage,
} from "./types";

const BASE = "/api";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ── Projects ──
export const api = {
  listProjects: () => jsonFetch<Project[]>(`${BASE}/projects`),
  createProject: (data: Partial<Project>) =>
    jsonFetch<Project>(`${BASE}/projects`, { method: "POST", body: JSON.stringify(data) }),
  getProject: (id: string) => jsonFetch<Project>(`${BASE}/projects/${id}`),
  deleteProject: (id: string) => jsonFetch<void>(`${BASE}/projects/${id}`, { method: "DELETE" }),

  // ── Datasets ──
  listDatasets: (projectId?: string) =>
    jsonFetch<Dataset[]>(`${BASE}/datasets${projectId ? `?project_id=${projectId}` : ""}`),
  getDataset: (id: string) => jsonFetch<Dataset>(`${BASE}/datasets/${id}`),
  createDataset: (data: Partial<Dataset> & { project_id: string }) =>
    jsonFetch<Dataset>(`${BASE}/datasets`, { method: "POST", body: JSON.stringify(data) }),
  deleteDataset: (id: string) => jsonFetch<void>(`${BASE}/datasets/${id}`, { method: "DELETE" }),

  listRows: (datasetId: string, skip = 0, limit = 500) =>
    jsonFetch<DatasetRow[]>(`${BASE}/datasets/${datasetId}/rows?skip=${skip}&limit=${limit}`),
  addRow: (datasetId: string, row: Partial<DatasetRow>) =>
    jsonFetch<DatasetRow>(`${BASE}/datasets/${datasetId}/rows`, {
      method: "POST",
      body: JSON.stringify(row),
    }),
  updateRow: (rowId: string, row: Partial<DatasetRow>) =>
    jsonFetch<DatasetRow>(`${BASE}/datasets/rows/${rowId}`, {
      method: "PUT",
      body: JSON.stringify(row),
    }),
  deleteRow: (rowId: string) =>
    jsonFetch<void>(`${BASE}/datasets/rows/${rowId}`, { method: "DELETE" }),

  exportDataset: async (datasetId: string, format: "alpaca" | "sharegpt" | "jsonl") => {
    const res = await fetch(`${BASE}/datasets/${datasetId}/export?format=${format}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error(await res.text());
    return res.blob();
  },

  // ── Chat / generation ──
  generateDataset: (data: {
    project_id: string;
    description: string;
    domain: string;
    num_rows: number;
    provider: string;
    model: string;
    seed_examples?: { instruction: string; output: string }[];
    dataset_name?: string;
  }) =>
    jsonFetch<Dataset>(`${BASE}/chat/generate-dataset`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  streamChat: async (
    body: {
      project_id?: string;
      messages: ChatMessage[];
      provider: string;
      model: string;
    },
    onDelta: (delta: string) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ) => {
    const res = await fetch(`${BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.body) {
      onError("No response stream");
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop() || "";
      for (const evt of events) {
        const lines = evt.split("\n");
        let event = "message";
        let data = "";
        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }
        if (event === "delta") onDelta(data);
        if (event === "done") onDone();
        if (event === "error") onError(data);
      }
    }
    onDone();
  },

  // ── Training ──
  listRuns: (projectId?: string) =>
    jsonFetch<TrainingRun[]>(
      `${BASE}/training/runs${projectId ? `?project_id=${projectId}` : ""}`,
    ),
  createRun: (data: {
    project_id: string;
    dataset_id?: string;
    name: string;
    base_model: string;
    hardware: "tpu" | "gpu";
    method?: "lora" | "qlora" | "full";
    config?: Record<string, unknown>;
  }) =>
    jsonFetch<TrainingRun>(`${BASE}/training/runs`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  getRun: (id: string) => jsonFetch<TrainingRun>(`${BASE}/training/runs/${id}`),
  downloadNotebook: (id: string) => `${BASE}/training/runs/${id}/notebook`,

  // ── Benchmarks ──
  listBenchmarks: (params?: { benchmark_type?: string; hardware?: string }) => {
    const q = new URLSearchParams(params as Record<string, string>).toString();
    return jsonFetch<BenchmarkResult[]>(`${BASE}/benchmarks/runs${q ? `?${q}` : ""}`);
  },
  createBenchmark: (data: Partial<BenchmarkResult>) =>
    jsonFetch<BenchmarkResult>(`${BASE}/benchmarks/runs`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  compareBenchmarks: (model?: string) =>
    jsonFetch<{ comparison: Record<string, BenchmarkResult>; count: number }>(
      `${BASE}/benchmarks/compare${model ? `?model=${encodeURIComponent(model)}` : ""}`,
    ),
};
