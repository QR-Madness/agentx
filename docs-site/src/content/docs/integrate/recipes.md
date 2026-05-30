# Recipes

Copy-paste starting points in `curl`, JavaScript, and Python. They assume the API at
`http://localhost:12319` with auth disabled — if auth is on, add an `X-Auth-Token` header (see
[Integration Overview](overview.md#authentication)).

## curl

```bash
# One-shot chat (JSON response)
curl -s -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you help me with?", "use_memory": true}'

# Streaming chat — -N disables buffering so events arrive live
curl -N -X POST http://localhost:12319/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Draft a release note", "model": "llama3.2"}'

# Translate (NLLB-200 language codes)
curl -s -X POST http://localhost:12319/api/tools/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "targetLanguage": "fra_Latn"}'

# Recall the user's history on a topic
curl -s -X POST http://localhost:12319/api/memory/user-history \
  -H "Content-Type: application/json" \
  -d '{"topic": "deployment", "limit": 5}'
```

## JavaScript

```js
const BASE = "http://localhost:12319/api";

// One-shot chat
async function chat(message) {
  const res = await fetch(`${BASE}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, use_memory: true }),
  });
  if (!res.ok) throw new Error((await res.json()).error);
  return (await res.json()).answer;
}

// Streaming chat — the endpoint is POST, so read the body stream (not EventSource)
async function chatStream(message, onToken) {
  const res = await fetch(`${BASE}/agent/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const event = frame.match(/^event: (.*)$/m)?.[1];
      const data = frame.match(/^data: (.*)$/m)?.[1];
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "chunk") onToken(payload.content);
    }
  }
}

// Re-attach to a detached run — this endpoint is GET, so EventSource works
function attach(runId, onToken) {
  const es = new EventSource(`${BASE}/agent/chat/stream/attach?run_id=${runId}`);
  es.addEventListener("chunk", (e) => onToken(JSON.parse(e.data).content));
  es.addEventListener("close", () => es.close());
}
```

## Python

```python
import httpx

BASE = "http://localhost:12319/api"

# One-shot chat
def chat(message: str) -> str:
    r = httpx.post(f"{BASE}/agent/chat", json={"message": message, "use_memory": True})
    r.raise_for_status()
    return r.json()["answer"]

# Streaming chat (SSE over a POST body)
def chat_stream(message: str):
    with httpx.stream("POST", f"{BASE}/agent/chat/stream", json={"message": message}) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event == "chunk":
                import json
                print(json.loads(line[len("data: "):])["content"], end="", flush=True)
```

## Next steps

- The full event list and detached-run flow: [Streaming & Detached Runs](streaming.md).
- Every endpoint with request/response examples: [API Reference](../api/endpoints.md).
