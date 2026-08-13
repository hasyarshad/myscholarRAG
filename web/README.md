# myScholar web

A Next.js demo of the chat widget, and the canonical copy of the widget itself.

## Run it

```bash
uv run myscholar serve                      # the API, on :8000
cd web && npm install && npm run dev        # the demo, on :3000
```

The demo defaults to `http://127.0.0.1:8000`. Override with `MYSCHOLAR_API`.

## Using the widget elsewhere

`components/scholar-chat/` is self-contained — five files, no dependencies
beyond React. Copy the directory into a portfolio and render it:

```tsx
import { ScholarChat } from "@/components/scholar-chat/ScholarChat";

<ScholarChat apiBase="/api/scholar" />
```

Styling lives in `app/globals.css` under `.scholar`, kept deliberately plain so
a host site's design system can replace it without touching the components.

A copy rather than an npm package: publishing and versioning for a single
consumer is overhead, and a private package would make the project harder for
someone else to run.

## What the widget shows

The stage indicator is the point. A visitor watching

```
Searching the papers      8 blocks           3.4s
Checking what was found   8 kept, 0 dropped  5.7s
Writing the answer        attempt 1         15.0s
Verifying every citation  2 citation issues 15.0s
Writing the answer        attempt 2         18.9s
Verifying every citation  citations valid   18.9s
```

is seeing the system reject its own draft — which is what makes it trustworthy,
and is invisible in the finished answer.

Citation chips expand to the paper, section and page range a claim came from.
Refusals name what the corpus *does* cover rather than being a dead end.

## Notes

- SSE is parsed from a `fetch` body rather than with `EventSource`, which is
  GET-only and cannot set headers — so it could not carry a session id or the
  shared secret the production proxy will inject.
- Partial frames are buffered across reads, because a network read can end
  mid-frame and parsing half a frame yields a JSON error rather than an event.
- The live region announces completion, not every stage: stage-level updates
  make a screen reader unusable.
- The abort controller fires on unmount, so the server releases its GPU slot
  rather than finishing an answer nobody will read.
