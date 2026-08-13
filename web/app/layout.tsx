import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "myScholar — ask about the research",
  description: "A citation-grounded assistant over 20 research papers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
