"use client";
import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, Download, Save, X, Edit, Cpu } from "lucide-react";
import Link from "next/link";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import type { DatasetRow } from "@/lib/types";

export default function DatasetEditorPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const qc = useQueryClient();

  const { data: dataset } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.getDataset(id),
  });
  const { data: rows = [] } = useQuery({
    queryKey: ["dataset-rows", id],
    queryFn: () => api.listRows(id, 0, 1000),
  });

  const [editing, setEditing] = useState<DatasetRow | null>(null);
  const [creating, setCreating] = useState(false);
  const [exportFormat, setExportFormat] = useState<"alpaca" | "sharegpt" | "jsonl">("alpaca");

  const update = useMutation({
    mutationFn: (row: DatasetRow) =>
      api.updateRow(row.id, {
        instruction: row.instruction,
        input: row.input,
        output: row.output,
        system_prompt: row.system_prompt,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dataset-rows", id] }),
  });
  const create = useMutation({
    mutationFn: (row: Partial<DatasetRow>) => api.addRow(id, row),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-rows", id] });
      qc.invalidateQueries({ queryKey: ["dataset", id] });
    },
  });
  const remove = useMutation({
    mutationFn: (rowId: string) => api.deleteRow(rowId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-rows", id] });
      qc.invalidateQueries({ queryKey: ["dataset", id] });
    },
  });

  const downloadExport = async () => {
    const blob = await api.exportDataset(id, exportFormat);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${dataset?.name?.replace(/\s+/g, "_") ?? "dataset"}_${exportFormat}.${
      exportFormat === "alpaca" ? "json" : "jsonl"
    }`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <Header
        title={dataset?.name ?? "Dataset"}
        subtitle={dataset ? `${dataset.row_count} rows · format ${dataset.format}` : undefined}
      />
      <div className="container py-6 max-w-6xl">
        <Card className="mb-4">
          <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
            <div className="flex items-center gap-3">
              <Badge variant="outline">{dataset?.format ?? "—"}</Badge>
              <Badge variant="secondary">{dataset?.source ?? "—"}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Select value={exportFormat} onValueChange={(v) => setExportFormat(v as never)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="alpaca">Alpaca</SelectItem>
                  <SelectItem value="sharegpt">ShareGPT</SelectItem>
                  <SelectItem value="jsonl">JSONL</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" onClick={downloadExport}>
                <Download className="mr-2 h-4 w-4" /> Export
              </Button>
              <Button asChild>
                <Link href={`/training?dataset_id=${id}`}>
                  <Cpu className="mr-2 h-4 w-4" /> Fine-tune
                </Link>
              </Button>
              <Button onClick={() => setCreating(true)}>
                <Plus className="mr-2 h-4 w-4" /> Add row
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Rows</CardTitle>
            <CardDescription>{dataset?.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Instruction</TableHead>
                  <TableHead>Input</TableHead>
                  <TableHead>Output</TableHead>
                  <TableHead className="w-24"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell className="text-muted-foreground">{r.position + 1}</TableCell>
                    <TableCell className="max-w-xs truncate">{r.instruction}</TableCell>
                    <TableCell className="max-w-xs truncate text-muted-foreground">
                      {r.input || "—"}
                    </TableCell>
                    <TableCell className="max-w-md truncate">{r.output}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Button size="icon" variant="ghost" onClick={() => setEditing(r)}>
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => remove.mutate(r.id)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <RowDialog
        open={!!editing}
        row={editing}
        onClose={() => setEditing(null)}
        onSave={(r) => {
          if (editing) update.mutate({ ...editing, ...r });
          setEditing(null);
        }}
      />
      <RowDialog
        open={creating}
        row={null}
        onClose={() => setCreating(false)}
        onSave={(r) => {
          create.mutate(r);
          setCreating(false);
        }}
      />
    </>
  );
}

function RowDialog({
  open,
  row,
  onClose,
  onSave,
}: {
  open: boolean;
  row: DatasetRow | null;
  onClose: () => void;
  onSave: (r: Partial<DatasetRow>) => void;
}) {
  const [instruction, setInstruction] = useState(row?.instruction ?? "");
  const [input, setInput] = useState(row?.input ?? "");
  const [output, setOutput] = useState(row?.output ?? "");
  const [systemPrompt, setSystemPrompt] = useState(row?.system_prompt ?? "");

  useEffect(() => {
    if (open) {
      setInstruction(row?.instruction ?? "");
      setInput(row?.input ?? "");
      setOutput(row?.output ?? "");
      setSystemPrompt(row?.system_prompt ?? "");
    }
  }, [open, row]);

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{row ? "Edit row" : "Add row"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Instruction</label>
            <Textarea
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              rows={2}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Input (optional)</label>
            <Textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">Output</label>
            <Textarea value={output} onChange={(e) => setOutput(e.target.value)} rows={4} />
          </div>
          <div className="space-y-1.5">
            <label className="text-sm font-medium">System prompt (optional)</label>
            <Input value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            <X className="mr-2 h-4 w-4" /> Cancel
          </Button>
          <Button
            onClick={() =>
              onSave({
                instruction,
                input: input || null,
                output,
                system_prompt: systemPrompt || null,
              })
            }
            disabled={!instruction.trim() || !output.trim()}
          >
            <Save className="mr-2 h-4 w-4" /> Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
