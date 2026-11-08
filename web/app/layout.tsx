import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "myScholar: ask about the research",
  description:
    "A citation-grounded assistant over Haseeb's research papers, CV and projects.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // Browser extensions add attributes to <html> before hydration. This silences
    // that element's own attributes only; mismatches inside the page still report.
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
