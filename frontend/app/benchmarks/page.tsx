"use client";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/utils";

interface Metrics {
  throughput_tokens_per_sec?: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  peak_memory_gb?: number;
  cost_per_1m_tokens_usd?: number;
  total_time_sec?: number;
}

export default function BenchmarksPage() {
  const { data: items = [] } = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api.listBenchmarks(),
  });

  const tpu = items.filter((b) => b.hardware.toLowerCase().includes("tpu"))[0];
  const gpu = items.filter(
    (b) =>
      b.hardware.toLowerCase().includes("gpu") ||
      /rtx|a100|h100|l4|t4/i.test(b.hardware),
  )[0];

  const chartData = (() => {
    if (!tpu && !gpu) return [];
    const tpuMetrics = (tpu?.metrics || {}) as Metrics;
    const gpuMetrics = (gpu?.metrics || {}) as Metrics;
    return [
      {
        metric: "Throughput (tok/s)",
        TPU: tpuMetrics.throughput_tokens_per_sec ?? 0,
        GPU: gpuMetrics.throughput_tokens_per_sec ?? 0,
      },
      {
        metric: "Latency p50 (ms)",
        TPU: tpuMetrics.latency_p50_ms ?? 0,
        GPU: gpuMetrics.latency_p50_ms ?? 0,
      },
      {
        metric: "Latency p95 (ms)",
        TPU: tpuMetrics.latency_p95_ms ?? 0,
        GPU: gpuMetrics.latency_p95_ms ?? 0,
      },
      {
        metric: "Peak mem (GB)",
        TPU: tpuMetrics.peak_memory_gb ?? 0,
        GPU: gpuMetrics.peak_memory_gb ?? 0,
      },
    ];
  })();

  return (
    <>
      <Header title="Benchmarks" subtitle="TPU vs GPU — vLLM throughput / latency / memory" />
      <div className="container py-6 max-w-6xl space-y-6">
        <div className="grid gap-4 md:grid-cols-2">
          <KpiCard label="TPU runs" value={items.filter((b) => b.hardware.toLowerCase().includes("tpu")).length} hint="Cloud TPU v5e-4" tone="tpu" />
          <KpiCard label="GPU runs" value={items.filter((b) => /gpu|rtx|a100|h100|l4|t4/i.test(b.hardware)).length} hint="CUDA + vLLM" tone="gpu" />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Latest TPU vs GPU comparison</CardTitle>
            <CardDescription>
              Pulls the most recent benchmark from each hardware class.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {chartData.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No benchmark results yet. Run{" "}
                <code className="text-xs">scripts/benchmark_vllm.py</code> on TPU and GPU
                and POST results to <code className="text-xs">/api/benchmarks/runs</code>.
              </p>
            ) : (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="TPU" fill="hsl(24 100% 50%)" />
                    <Bar dataKey="GPU" fill="hsl(140 70% 45%)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">All benchmark runs</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>HW</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Throughput</TableHead>
                  <TableHead>p95 latency</TableHead>
                  <TableHead>Mem</TableHead>
                  <TableHead>Cost / 1M</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((b) => {
                  const m = (b.metrics || {}) as Metrics;
                  return (
                    <TableRow key={b.id}>
                      <TableCell className="font-medium">{b.name}</TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            b.hardware.toLowerCase().includes("tpu") ? "tpu" : "gpu"
                          }
                        >
                          {b.hardware}
                        </Badge>
                      </TableCell>
                      <TableCell>{b.benchmark_type}</TableCell>
                      <TableCell className="font-mono text-xs">{b.model}</TableCell>
                      <TableCell>{formatNumber(m.throughput_tokens_per_sec ?? 0)}</TableCell>
                      <TableCell>{formatNumber(m.latency_p95_ms ?? 0)}</TableCell>
                      <TableCell>{formatNumber(m.peak_memory_gb ?? 0)}</TableCell>
                      <TableCell>${formatNumber(m.cost_per_1m_tokens_usd ?? 0)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function KpiCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: number;
  hint: string;
  tone: "tpu" | "gpu";
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-3xl font-bold">{value}</div>
            <div className="text-sm text-muted-foreground">{label}</div>
          </div>
          <Badge variant={tone}>{hint}</Badge>
        </div>
      </CardContent>
    </Card>
  );
}
