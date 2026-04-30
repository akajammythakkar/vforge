"use client";
import { useState } from "react";
import { Save, AlertTriangle } from "lucide-react";

import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

export default function SettingsPage() {
  const [touched, setTouched] = useState(false);
  return (
    <>
      <Header title="Settings" subtitle="Provider keys, defaults, GCP" />
      <div className="container py-6 max-w-3xl space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Where to put your keys</CardTitle>
            <CardDescription>
              vForge reads provider credentials from environment variables. Edit{" "}
              <code className="text-xs">.env</code> at the repo root.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-md bg-muted p-4 text-xs">
{`HF_TOKEN=...
TOGETHER_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
GCP_PROJECT_ID=...
TPU_ZONE=us-central1-a
TPU_TYPE=v5e-4`}
            </pre>
          </CardContent>
        </Card>

        <Tabs defaultValue="providers">
          <TabsList>
            <TabsTrigger value="providers">Providers</TabsTrigger>
            <TabsTrigger value="storage">Storage</TabsTrigger>
            <TabsTrigger value="gcp">Google Cloud</TabsTrigger>
          </TabsList>

          <TabsContent value="providers" className="space-y-4">
            <Section title="Tier 1 — open weights" tone="safe">
              <Field label="OLLAMA_BASE_URL" placeholder="http://localhost:11434" />
              <Field label="TOGETHER_API_KEY" placeholder="tgp_..." secret />
            </Section>
            <Section title="Tier 2 — proprietary (check ToS)" tone="warn">
              <Field label="OPENAI_API_KEY" placeholder="sk-..." secret />
              <Field label="ANTHROPIC_API_KEY" placeholder="sk-ant-..." secret />
              <Field label="GOOGLE_API_KEY" placeholder="AIza..." secret />
            </Section>
          </TabsContent>

          <TabsContent value="storage" className="space-y-4">
            <Section title="HuggingFace Hub" tone="safe">
              <Field label="HF_TOKEN" placeholder="hf_..." secret />
              <Field label="HF_DEFAULT_ORG" placeholder="your-org" />
            </Section>
            <Section title="Google Cloud Storage" tone="safe">
              <Field label="GCS_BUCKET" placeholder="gs://your-bucket" />
            </Section>
          </TabsContent>

          <TabsContent value="gcp" className="space-y-4">
            <Section title="Cloud TPU" tone="safe">
              <Field label="GCP_PROJECT_ID" placeholder="my-project-123" />
              <Field label="TPU_ZONE" placeholder="us-central1-a" />
              <Field label="TPU_TYPE" placeholder="v5e-4" />
            </Section>
          </TabsContent>
        </Tabs>

        {touched && (
          <Card className="border-amber-500/50">
            <CardContent className="flex gap-3 py-4">
              <AlertTriangle className="h-5 w-5 text-amber-500 shrink-0" />
              <p className="text-sm">
                These fields are read-only in MVP — they reflect your{" "}
                <code className="text-xs">.env</code> file. Restart the backend to apply
                changes.
              </p>
            </CardContent>
          </Card>
        )}

        <Button onClick={() => setTouched(true)}>
          <Save className="mr-2 h-4 w-4" /> Reload from .env
        </Button>
      </div>
    </>
  );
}

function Section({
  title,
  tone,
  children,
}: {
  title: string;
  tone: "safe" | "warn";
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{title}</CardTitle>
          <Badge variant={tone === "safe" ? "secondary" : "destructive"}>
            {tone === "safe" ? "open / safe" : "check ToS"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">{children}</CardContent>
    </Card>
  );
}

function Field({ label, placeholder, secret }: { label: string; placeholder: string; secret?: boolean }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-mono">{label}</Label>
      <Input type={secret ? "password" : "text"} placeholder={placeholder} disabled />
    </div>
  );
}
