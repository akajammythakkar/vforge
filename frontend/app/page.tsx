"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, MessageSquare, Database, Cpu, BarChart3 } from "lucide-react";
import { Header } from "@/components/layout/header";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

const steps = [
  {
    title: "1. Discover",
    description: "Chat with vForge to design your dataset. The assistant asks the right questions.",
    href: "/chat",
    icon: MessageSquare,
  },
  {
    title: "2. Generate",
    description: "Synthesize instruction/response pairs with open-weight LLMs. Edit them in a table.",
    href: "/datasets",
    icon: Database,
  },
  {
    title: "3. Fine-tune",
    description: "Run LoRA on Cloud TPU and on GPU. One generated Colab notebook per target.",
    href: "/training",
    icon: Cpu,
  },
  {
    title: "4. Benchmark",
    description: "Compare TPU vs GPU on throughput, latency, memory, and cost — with vLLM.",
    href: "/benchmarks",
    icon: BarChart3,
  },
];

export default function Dashboard() {
  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => api.listDatasets(),
  });
  const { data: runs } = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns(),
  });
  const { data: benchmarks } = useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => api.listBenchmarks(),
  });

  return (
    <>
      <Header title="Dashboard" subtitle="Sprint Q1 2026 — vForge" />
      <div className="container py-8 max-w-6xl">
        <section className="mb-10">
          <Badge variant="tpu" className="mb-3">
            Google TPU Sprint · Q1 2026
          </Badge>
          <h1 className="text-3xl font-bold tracking-tight">
            From idea to TPU-trained model in four steps.
          </h1>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            vForge generates synthetic instruction-tuning datasets from a chat,
            fine-tunes an open model on Cloud TPU and on GPU, and benchmarks both with
            vLLM. The whole pipeline is reproducible and open source.
          </p>
        </section>

        <section className="mb-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Projects" value={projects?.length ?? 0} />
          <StatCard label="Datasets" value={datasets?.length ?? 0} />
          <StatCard label="Training runs" value={runs?.length ?? 0} />
          <StatCard label="Benchmarks" value={benchmarks?.length ?? 0} />
        </section>

        <section className="grid gap-4 md:grid-cols-2">
          {steps.map((s) => {
            const Icon = s.icon;
            return (
              <Card key={s.href}>
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="rounded-md bg-primary/10 p-2 text-primary">
                      <Icon className="h-5 w-5" />
                    </div>
                    <CardTitle>{s.title}</CardTitle>
                  </div>
                  <CardDescription className="pt-2">{s.description}</CardDescription>
                </CardHeader>
                <CardFooter>
                  <Button asChild variant="ghost" className="ml-auto">
                    <Link href={s.href}>
                      Open <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </section>
      </div>
    </>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="text-3xl font-bold">{value}</div>
        <div className="text-sm text-muted-foreground">{label}</div>
      </CardContent>
    </Card>
  );
}
