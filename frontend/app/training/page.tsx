"use client";
import { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Cpu, Play, Download, Cloud, Zap } from "lucide-react";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";

const BASE_MODELS = [
  "google/gemma-4-E2B-it",
  "google/gemma-4-E4B-it",
  "google/gemma-4-26B-A4B-it",
  "meta-llama/Llama-3.2-1B",
  "meta-llama/Llama-3.2-3B",
  "Qwen/Qwen2.5-1.5B",
  "Qwen/Qwen2.5-7B",
  "mistralai/Mistral-7B-v0.3",
];

export default function TrainingPage() {
  const search = useSearchParams();
  const datasetIdFromQuery = search.get("dataset_id") ?? "";
  const qc = useQueryClient();

  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.listDatasets(),
  });
  const { data: runs } = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns(),
  });
  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  const [name, setName] = useState("gemma-4-e2b-lora-tpu");
  const [baseModel, setBaseModel] = useState("google/gemma-4-E2B-it");
  const [hardware, setHardware] = useState<"tpu" | "gpu">("tpu");
  const [datasetId, setDatasetId] = useState(datasetIdFromQuery);
  const [rank, setRank] = useState(8);
  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(4);
  const [lr, setLr] = useState("1e-4");

  const create = useMutation({
    mutationFn: () => {
      const dataset = datasets?.find((d) => d.id === datasetId);
      const project_id = dataset?.project_id || projects?.[0]?.id;
      if (!project_id) throw new Error("No project found. Create a dataset first.");
      return api.createRun({
        project_id,
        dataset_id: datasetId || undefined,
        name,
        base_model: baseModel,
        hardware,
        method: "lora",
        config: {
          rank,
          alpha: rank * 2,
          epochs,
          batch_size: batchSize,
          lr,
          max_seq_len: 1024,
        },
      });
    },
    onSuccess: (run) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      window.open(api.downloadNotebook(run.id), "_blank");
    },
  });

  return (
    <>
      <Header title="Fine-tuning" subtitle="Configure a run and download a Colab notebook" />
      <div className="container py-6 max-w-6xl grid gap-6 lg:grid-cols-[1fr_400px]">
        <Card>
          <CardHeader>
            <CardTitle>Configure training run</CardTitle>
            <CardDescription>
              Pick a base model + hardware. We generate a Colab notebook that runs the
              corresponding script (`scripts/finetune_tpu.py` or `finetune_gpu.py`).
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5 sm:col-span-2">
              <Label>Run name</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} />
            </div>

            <div className="space-y-1.5">
              <Label>Base model</Label>
              <Select value={baseModel} onValueChange={setBaseModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BASE_MODELS.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Hardware</Label>
              <Select value={hardware} onValueChange={(v) => setHardware(v as "tpu" | "gpu")}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tpu">Cloud TPU v5e-4 (JAX/Keras 3.0)</SelectItem>
                  <SelectItem value="gpu">CUDA GPU (PyTorch + PEFT)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5 sm:col-span-2">
              <Label>Dataset</Label>
              <Select value={datasetId} onValueChange={setDatasetId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent>
                  {(datasets ?? []).map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.name} ({d.row_count} rows)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>LoRA rank</Label>
              <Input
                type="number"
                value={rank}
                onChange={(e) => setRank(parseInt(e.target.value) || 8)}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Epochs</Label>
              <Input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Batch size</Label>
              <Input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 4)}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Learning rate</Label>
              <Input value={lr} onChange={(e) => setLr(e.target.value)} />
            </div>

            <div className="sm:col-span-2 flex gap-2 pt-2">
              <Button
                onClick={() => create.mutate()}
                disabled={create.isPending}
                size="lg"
                className="flex-1"
              >
                {hardware === "tpu" ? <Cloud className="mr-2 h-4 w-4" /> : <Zap className="mr-2 h-4 w-4" />}
                {create.isPending ? "Creating..." : "Create run + download notebook"}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent runs</CardTitle>
            <CardDescription>{runs?.length ?? 0} total</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {(runs ?? []).slice(0, 12).map((r) => (
              <Link
                key={r.id}
                href={`/training/${r.id}`}
                className="block rounded-md border p-3 transition hover:bg-accent"
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium text-sm truncate">{r.name}</div>
                  <Badge variant={r.hardware === "tpu" ? "tpu" : "gpu"}>{r.hardware.toUpperCase()}</Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">{r.base_model}</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {r.status} · {new Date(r.created_at).toLocaleString()}
                </div>
              </Link>
            ))}
            {(runs ?? []).length === 0 && (
              <p className="text-sm text-muted-foreground">No runs yet. Configure one →</p>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
