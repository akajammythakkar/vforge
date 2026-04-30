"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Plus, Database, ChevronRight } from "lucide-react";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function DatasetsPage() {
  const { data: datasets, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.listDatasets(),
  });

  return (
    <>
      <Header title="Datasets" subtitle="Generated and uploaded fine-tuning datasets" />
      <div className="container py-6 max-w-6xl">
        <div className="mb-6 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {datasets ? `${datasets.length} dataset${datasets.length !== 1 ? "s" : ""}` : "—"}
          </p>
          <Button asChild>
            <Link href="/chat">
              <Plus className="mr-2 h-4 w-4" />
              New from chat
            </Link>
          </Button>
        </div>

        {isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {datasets && datasets.length === 0 && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-16 text-center">
              <Database className="mb-4 h-12 w-12 text-muted-foreground" />
              <h3 className="text-lg font-semibold">No datasets yet</h3>
              <p className="mb-4 max-w-md text-sm text-muted-foreground">
                Start a chat with vForge to design and generate your first instruction-tuning dataset.
              </p>
              <Button asChild>
                <Link href="/chat">Start chat</Link>
              </Button>
            </CardContent>
          </Card>
        )}

        <div className="grid gap-4 md:grid-cols-2">
          {(datasets ?? []).map((d) => (
            <Card key={d.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{d.name}</CardTitle>
                  <Badge variant="outline">{d.format}</Badge>
                </div>
                <CardDescription>
                  {d.row_count} rows · {new Date(d.created_at).toLocaleString()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="line-clamp-2 text-sm text-muted-foreground">
                  {d.description || "—"}
                </p>
                <Button variant="ghost" asChild className="mt-3 ml-auto block w-fit">
                  <Link href={`/datasets/${d.id}`}>
                    Open <ChevronRight className="ml-1 inline h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </>
  );
}
