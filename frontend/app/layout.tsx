import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { Providers } from "@/components/layout/providers";

export const metadata: Metadata = {
  title: "vForge — TPU vs GPU LLM fine-tuning",
  description:
    "Open-source platform for LLM fine-tuning benchmarking — TPU vs GPU — with conversational dataset generation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background antialiased">
        <Providers>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="flex-1 overflow-x-hidden">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
