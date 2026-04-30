import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, digits = 2): string {
  if (n === undefined || n === null || isNaN(n)) return "—";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(digits) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(digits) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(digits) + "k";
  return n.toFixed(digits);
}
