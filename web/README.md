# myScholar web

A Next.js demo of the chat widget, and the canonical copy of the widget itself.

## Run it

```bash
uv run myscholar serve                      # the API, on :8000
cd web && npm install && npm run dev        # the demo, on :3000
```

The widget calls the same-origin proxy in `app/api/scholar/`, which forwards to
`MYSCHOLAR_API_URL` (default `http://127.0.0.1:8000`) and adds
`MYSCHOLAR_API_SECRET` when it is set.

## Using the widget elsewhere

`components/scholar-chat/` is self-contained: six files, no dependencies beyond
React. Copy it, and the proxy route, into a Next.js site and render it:

```tsx
import { ScholarChat } from "@/components/scholar-chat/ScholarChat";

<ScholarChat apiBase="/api/scholar" />
```

Styles are in `scholar-chat.css`, scoped under `.scholar`. Colours come from
CSS custom properties (`--scholar-accent` and others) that a host page can override.

## What the widget shows

Each stage of the pipeline appears as it runs, with what the server found:

```
Reading the question                                      0.3s
Searching                 8 passages from 3 papers        2.1s
Writing the answer        1840 chars                      9.6s
Verifying every citation  dropped 1 uncited sentence(s)   9.6s
Verifying every citation  grounded                       11.2s
Done                                                     11.2s
```

Citation chips expand to the paper, section and page range a claim came from.
Refusals list the closest topics the corpus does cover.

## Notes

- SSE is parsed from a `fetch` body because `EventSource` is GET-only and
  cannot send a request body or headers.
- Partial frames are buffered across reads, since a network read can end mid-frame.
- The live region announces completion, not every stage, which would make a
  screen reader unusable.
- The request is aborted on unmount, so the server can stop work on an answer
  nobody will read.
