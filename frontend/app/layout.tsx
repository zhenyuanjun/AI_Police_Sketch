import "./globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = { title: "AI Police Sketch", description: "Iterative facial sketch research prototype" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }

