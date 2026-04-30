"use client";
import { Github, Flame } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="flex h-16 items-center justify-between border-b bg-background px-6">
      <div className="flex items-center gap-3 md:hidden">
        <Flame className="h-5 w-5 text-primary" />
        <span className="font-bold">vForge</span>
      </div>
      <div className="hidden md:block">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {subtitle && (
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" asChild>
          <a
            href="https://github.com/"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub"
          >
            <Github className="h-4 w-4" />
          </a>
        </Button>
      </div>
    </header>
  );
}
