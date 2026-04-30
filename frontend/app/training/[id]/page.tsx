"use client";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Download, Cpu, ArrowLeft } from "lucide-react";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function TrainingRunPage() {
  const { id } = useParams<{ id: string }>();
  const { data: run } = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
  });

  if (!run) return null;
  const cfg = (run.config || {}) as Record<string, unknown>;
  const metrics = (run.metrics || {}) as Record<string, unknown>;

  return (
    <>
      <Header title={run.name} subtitle={`${run.base_model} on ${run.hardware.toUpperCase()}`} />
      <div className="container py-6 max-w-6xl space-y-6">
        <div className="flex items-center justify-between">
          <Button variant="ghost" asChild>
            <Link href="/training">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back
            </Link>
          </Button>
          <div className="flex gap-2">
            <Badge variant={run.hardware === "tpu" ? "tpu" : "gpu"}>{run.hardware.toUpperCase()}</Badge>
            <Badge variant="outline">{run.method}</Badge>
            <Badge variant="secondary">{run.status}</Badge>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Notebook</CardTitle>
            <CardDescription>
              Download and open in Colab. Connect to a {run.hardware.toUpperCase()} runtime
              {run.hardware === "tpu" ? " (TPU v5e-4 ideal)" : ""} and run all cells.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild>
              <a href={api.downloadNotebook(run.id)} target="_blank" rel="noreferrer">
                <Download className="mr-2 h-4 w-4" />
                Download {run.hardware}_{run.id.slice(0, 6)}.ipynb
              </a>
            </Button>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Config</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(cfg).map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd className="font-mono">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Metrics</CardTitle>
              <CardDescription>
                Updated after the notebook reports back via{" "}
                <code className="text-xs">PATCH /api/training/runs/{"{id}"}/metrics</code>.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {Object.keys(metrics).length === 0 ? (
                <p className="text-sm text-muted-foreground">No metrics yet.</p>
              ) : (
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(metrics).map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-muted-foreground">{k}</dt>
                      <dd className="font-mono">{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
