"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Send, Sparkles, Loader2, AlertTriangle } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import type { ChatMessage, Project } from "@/lib/types";

const PROVIDERS = [
  { value: "ollama", label: "Ollama (local)", tier: 1 },
  { value: "together", label: "Together AI", tier: 1 },
  { value: "openai", label: "OpenAI", tier: 2 },
  { value: "anthropic", label: "Anthropic", tier: 2 },
  { value: "google", label: "Google Gemini", tier: 2 },
  { value: "custom", label: "Custom OpenAI-compatible", tier: 3 },
];

const MODELS_BY_PROVIDER: Record<string, string[]> = {
  ollama: ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b"],
  together: [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "google/gemma-2-27b-it",
  ],
  openai: ["gpt-4o-mini", "gpt-4o"],
  anthropic: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
  google: ["gemini-1.5-pro", "gemini-1.5-flash"],
  custom: [""],
};

export default function ChatPage() {
  const router = useRouter();
  const [provider, setProvider] = useState("ollama");
  const [model, setModel] = useState("llama3.1:8b");
  const [projectId, setProjectId] = useState<string>("");
  const [domain, setDomain] = useState("code");
  const [numRows, setNumRows] = useState(50);
  const [datasetName, setDatasetName] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm vForge. Tell me what you'd like to fine-tune a model to do, and I'll help design a training dataset. (e.g. 'a code-completion assistant for Pandas one-liners')",
    },
  ]);
  const [draft, setDraft] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const { data: projects, refetch: refetchProjects } = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  useEffect(() => {
    setModel(MODELS_BY_PROVIDER[provider]?.[0] || "");
  }, [provider]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const ensureProject = async (): Promise<Project> => {
    if (projectId) {
      const found = (projects ?? []).find((p) => p.id === projectId);
      if (found) return found;
    }
    const p = await api.createProject({
      name: datasetName || `${domain} dataset`,
      domain,
    });
    await refetchProjects();
    setProjectId(p.id);
    return p;
  };

  const send = async () => {
    if (!draft.trim() || isStreaming) return;
    setError(null);
    const next: ChatMessage[] = [...messages, { role: "user", content: draft }];
    setMessages(next);
    setDraft("");
    setIsStreaming(true);

    let assistant = "";
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    await api.streamChat(
      { messages: next, provider, model },
      (delta) => {
        assistant += delta;
        setMessages((m) => {
          const copy = [...m];
          copy[copy.length - 1] = { role: "assistant", content: assistant };
          return copy;
        });
      },
      () => setIsStreaming(false),
      (err) => {
        setError(err);
        setIsStreaming(false);
      },
    );
  };

  const generateDataset = async () => {
    setError(null);
    setIsGenerating(true);
    try {
      const project = await ensureProject();
      const description = messages
        .filter((m) => m.role === "user")
        .map((m) => m.content)
        .join("\n\n") || `Dataset for ${domain}.`;
      const ds = await api.generateDataset({
        project_id: project.id,
        description,
        domain,
        num_rows: numRows,
        provider,
        model,
        dataset_name: datasetName || undefined,
      });
      router.push(`/datasets/${ds.id}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  const tier = PROVIDERS.find((p) => p.value === provider)?.tier ?? 1;

  return (
    <>
      <Header title="Conversational dataset builder" subtitle="Discover intent, then generate" />
      <div className="container py-6 max-w-6xl grid gap-6 lg:grid-cols-[1fr_320px]">
        <Card className="flex flex-col h-[calc(100vh-9rem)]">
          <CardHeader className="border-b">
            <CardTitle className="text-base">Chat</CardTitle>
            <CardDescription>
              Describe your goal. The assistant will ask clarifying questions.
            </CardDescription>
          </CardHeader>
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
                    m.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  {m.content || (isStreaming && i === messages.length - 1 ? "…" : "")}
                </div>
              </div>
            ))}
            {error && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                <AlertTriangle className="inline mr-2 h-4 w-4" />
                {error}
              </div>
            )}
          </div>
          <div className="border-t p-4">
            <div className="flex gap-2">
              <Textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="Describe your fine-tuning goal..."
                rows={2}
                className="resize-none"
              />
              <Button onClick={send} disabled={isStreaming || !draft.trim()}>
                {isStreaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Provider</CardTitle>
              <CardDescription className="flex items-center gap-2">
                Tier {tier}
                {tier === 1 && <Badge variant="secondary">safe / open</Badge>}
                {tier === 2 && <Badge variant="destructive">check ToS</Badge>}
                {tier === 3 && <Badge variant="outline">custom</Badge>}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label>Provider</Label>
                <Select value={provider} onValueChange={setProvider}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PROVIDERS.map((p) => (
                      <SelectItem key={p.value} value={p.value}>
                        {p.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Model</Label>
                {provider === "custom" ? (
                  <Input value={model} onChange={(e) => setModel(e.target.value)} />
                ) : (
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(MODELS_BY_PROVIDER[provider] || []).map((m) => (
                        <SelectItem key={m} value={m}>
                          {m}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              {tier === 2 && (
                <p className="text-xs text-muted-foreground">
                  Some providers' terms restrict using outputs to train competing models.
                  Check before publishing the dataset.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Generate dataset</CardTitle>
              <CardDescription>Synthesize rows from this conversation.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label>Project</Label>
                <Select value={projectId} onValueChange={setProjectId}>
                  <SelectTrigger>
                    <SelectValue placeholder="New project" />
                  </SelectTrigger>
                  <SelectContent>
                    {(projects ?? []).map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label>Dataset name</Label>
                <Input
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="my-coding-helper-v1"
                />
              </div>
              <div className="space-y-1.5">
                <Label>Domain</Label>
                <Input value={domain} onChange={(e) => setDomain(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Number of rows</Label>
                <Input
                  type="number"
                  min={1}
                  max={5000}
                  value={numRows}
                  onChange={(e) => setNumRows(parseInt(e.target.value) || 50)}
                />
              </div>
              <Button className="w-full" onClick={generateDataset} disabled={isGenerating}>
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating…
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" /> Generate dataset
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
