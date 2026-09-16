import { splitSseRecords, extractSseDataPayload } from '$lib/utils/sse';

export function errorText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object' && 'message' in value) return String(value.message);
  return 'Request failed';
}

export async function streamChat(body: object, signal: AbortSignal, event: (data: any) => void) {
  const response = await fetch('./v1/chat/completions', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal
  });
  if (!response.ok) {
    const text = await response.text();
    let detail;
    try { detail = errorText(JSON.parse(text).error); } catch { detail = text || response.statusText; }
    throw new Error(detail);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Empty response');
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) throw new Error('Connection interrupted. Send a new message to continue.');
      buffer += decoder.decode(value, {stream: true}).replace(/\r\n/g, '\n');
      const parsed = splitSseRecords(buffer);
      buffer = parsed.rest;
      for (const record of parsed.records) {
        const payload = extractSseDataPayload(record);
        if (!payload) continue;
        if (payload === '[DONE]') return;
        const data = JSON.parse(payload);
        if (data.error) throw new Error(errorText(data.error));
        event(data);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
