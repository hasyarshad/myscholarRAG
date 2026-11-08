/**
 * Same-origin proxy to the RAG backend: `/api/scholar/v1/ask` -> `$MYSCHOLAR_API_URL/v1/ask`.
 *
 * The browser sees one origin (no CORS), never learns the backend URL, and never
 * sees the shared secret, which is added here. Keep the secret out of any
 * `NEXT_PUBLIC_` variable, which would ship it in the client bundle.
 */

import { NextRequest } from "next/server";

// Node runtime, passing the stream through untouched. force-dynamic stops Next
// caching an SSE response.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.MYSCHOLAR_API_URL ?? "http://127.0.0.1:8000";
const SECRET = process.env.MYSCHOLAR_API_SECRET ?? "";

/** A container scaled to zero can take ~50s to wake. */
const TIMEOUT_MS = 90_000;

async function proxy(request: NextRequest, path: string[]): Promise<Response> {
  const target = new URL(`/${path.join("/")}`, BACKEND);
  target.search = request.nextUrl.search;

  const headers = new Headers({ accept: request.headers.get("accept") ?? "*/*" });
  if (request.headers.has("content-type")) {
    headers.set("content-type", request.headers.get("content-type")!);
  }
  if (SECRET) {
    headers.set("x-myscholar-secret", SECRET);
  }
  // The backend rate-limits per visitor, so pass on the original address.
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    headers.set("x-forwarded-for", forwarded);
  }

  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" ? undefined : await request.text(),
      signal: abort.signal,
      cache: "no-store",
    });

    const out = new Headers();
    for (const name of ["content-type", "cache-control"]) {
      const value = upstream.headers.get(name);
      if (value) out.set(name, value);
    }
    // Stops intermediate proxies (nginx and similar) buffering the stream.
    out.set("x-accel-buffering", "no");

    return new Response(upstream.body, { status: upstream.status, headers: out });
  } catch (error) {
    const asleep = error instanceof Error && error.name === "AbortError";
    return Response.json(
      {
        detail: asleep
          ? "The assistant is waking up (free hosting sleeps when idle). Try again in a moment."
          : "The assistant is unreachable.",
      },
      { status: 503 },
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, (await context.params).path);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, (await context.params).path);
}
