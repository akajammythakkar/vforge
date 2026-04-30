"use client";
import { useQuery } from "@tanstack/react-query";
import { Boxes, ExternalLink } from "lucide-react";
import Link from "next/link";

import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function ModelsPage() {
  const { data: runs = [] } = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns(),
  });

  const trained = runs.filter((r) => r.status === "done" || r.artifact_uri);

  return (
    <>
      <Header title="Models" subtitle="Trained adapters & exports" />
      <div className="container py-6 max-w-6xl">
        {trained.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-16 text-center">
              <Boxes className="mb-4 h-12 w-12 text-muted-foreground" />
              <h3 className="text-lg font-semibold">No trained models yet</h3>
              <p className="mb-4 max-w-md text-sm text-muted-foreground">
                Run a fine-tune on TPU or GPU. When it finishes, the trained adapter shows up here
                and can be pushed to HuggingFace, GCS, or Drive.
              </p>
              <Button asChild>
                <Link href="/training">Configure training</Link>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {trained.map((r) => (
              <Card key={r.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{r.name}</CardTitle>
                    <Badge variant={r.hardware === "tpu" ? "tpu" : "gpu"}>
                      {r.hardware.toUpperCase()}
                    </Badge>
                  </div>
                  <CardDescription>
                    {r.base_model} · {r.method.toUpperCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground mb-3">
                    {r.artifact_uri ?? "(no artifact uri set)"}
                  </div>
                  <Button variant="outline" asChild>
                    <Link href={`/training/${r.id}`}>
                      Open <ExternalLink className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
