import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "myScholar — ask about the research",
  description:
    "A citation-grounded assistant over Haseeb's research papers, CV and projects.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning on <html> only.
    //
    // Browser extensions modify this element before React hydrates — QuillBot
    // adds `data-qb-installed`, password managers and theme switchers do
    // similar — and React then reports a mismatch between the server HTML and
    // what it finds in the DOM. Nothing in the application is at fault and
    // nothing is broken, but the warning is noisy and trains you to ignore
    // hydration errors, which are worth reading when they are real.
    //
    // It suppresses the warning for this element's own attributes and does not
    // extend to children, so a genuine mismatch inside the page still reports.
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
