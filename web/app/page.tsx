import { ScholarChat } from "@/components/scholar-chat/ScholarChat";

export default function Home() {
  return (
    <main>
      <h1>myScholar</h1>
      <p className="lede">
        A retrieval-augmented assistant over 20 research papers. Every factual
        sentence carries a citation that has been verified against the passage
        it came from — and when the papers do not cover something, it says so
        rather than inventing an answer.
      </p>
      <ScholarChat />
    </main>
  );
}
