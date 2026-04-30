"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquare,
  Database,
  Cpu,
  BarChart3,
  Boxes,
  Settings,
  Flame,
} from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/training", label: "Training", icon: Cpu },
  { href: "/benchmarks", label: "Benchmarks", icon: BarChart3 },
  { href: "/models", label: "Models", icon: Boxes },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden md:flex md:w-60 md:flex-col md:border-r md:bg-card">
      <div className="flex h-16 items-center gap-2 border-b px-6">
        <Flame className="h-6 w-6 text-primary" />
        <div>
          <div className="text-lg font-bold tracking-tight">vForge</div>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
            TPU Sprint Q1 2026
          </div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 p-4">
        {nav.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t p-4 text-xs text-muted-foreground">
        <div className="font-medium text-foreground">Sprint deadline</div>
        <div>April 30, 2026</div>
      </div>
    </aside>
  );
}
