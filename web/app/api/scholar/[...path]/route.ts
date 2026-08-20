/**
 * Same-origin proxy to the RAG backend.
 *
 * The widget calls `/api/scholar/v1/ask`, this forwards it. Four things come
 * out of that indirection, and only the first is obvious:
 *
 *  1. **No CORS.** The browser sees one origin. The whole class of "widget says
 *     it cannot reach the assistant while the API is perfectly healthy" bugs
 *     disappears, because there is no preflight to misconfigure.
 *
 *  2. **The shared secret stays server-side.** CORS is a browser-only control
 *     and stops nothing from `curl`. The backend requires a secret; it is added
 *     here, in a server runtime, where the browser can never read it. Putting
 *     it in the client bundle — which `NEXT_PUBLIC_` would do — would be the
 *     same as having no secret at all.
 *
 *  3. **The backend URL is not public.** Nobody gets to point a load generator
 *     straight at the free-tier container.
 *
 *  4. **A cold start can be handled rather than shown.** Free hosts sleep. The
 *     first request after that takes tens of seconds, and this is the only
 *     place that knows enough to say so instead of failing blankly.
 */

import { NextRequest } from "next/server";

// Node, not edge: the streaming body is passed through untouched, and this must
// not be statically optimised or Next will try to cache an SSE response.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.MYSCHOLAR_API_URL ?? "http://127.0.0.1:8000";
const SECRET = process.env.MYSCHOLAR_API_SECRET ?? "";

/** Generous, because a sleeping free-tier container can take ~50s to wake. */
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
  // The backend rate-limits per visitor, and every request from here shares this
  // server's address. Without the original address it would see one client.
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
      // Streamed. Buffering here would defeat the point of server-sent events:
      // the answer would arrive complete after ten seconds of nothing.
      cache: "no-store",
    });

    const out = new Headers();
    for (const name of ["content-type", "cache-control"]) {
      const value = upstream.headers.get(name);
      if (value) out.set(name, value);
    }
    // Some proxies buffer streamed responses into oblivion unless told not to.
    out.set("x-accel-buffering", "no");

    return new Response(upstream.body, { status: upstream.status, headers: out });
  } catch (error) {
    const asleep = error instanceof Error && error.name === "AbortError";
    return Response.json(
      {
        detail: asleep
          ? "The assistant is waking up — free hosting sleeps when idle. Try again in a moment."
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
