// Read a mjlog from a local file or a tenhou.net URL and return its XML text,
// transparently decompressing gzip (find.cgi serves the raw gzipped log).

/** True when the source is a URL rather than a local path. */
export function isUrl(source: string): boolean {
  return /^https?:\/\//.test(source);
}

/**
 * Rewrite a tenhou.net replay-viewer URL (e.g. https://tenhou.net/0/?log=<id>&tw=1)
 * to the raw log download endpoint (/0/log/find.cgi?log=<id>&tw=N), so pasted
 * replay links just work. find.cgi requires `tw` (the player-view seat; any
 * seat serves the same log), so it is carried over — or defaulted to 0.
 * URLs already under /log/ — and non-tenhou URLs — pass through.
 */
export function normalizeUrl(url: string): string {
  const u = new URL(url);
  const id = u.searchParams.get("log");
  if (u.hostname.endsWith("tenhou.net") && id && !u.pathname.includes("/log/")) {
    const tw = u.searchParams.get("tw") ?? "0";
    return `https://tenhou.net/0/log/find.cgi?log=${id}&tw=${tw}`;
  }
  return url;
}

/** True if the bytes start with the gzip magic (0x1f 0x8b). */
function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  // copy into a fresh ArrayBuffer-backed view to satisfy BlobPart typing
  const part = new Uint8Array(bytes);
  const stream = new Blob([part]).stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function loadXml(source: string): Promise<string> {
  const bytes = isUrl(source)
    ? await fetchBytes(normalizeUrl(source))
    : await Deno.readFile(source);
  const raw = isGzip(bytes) ? await gunzip(bytes) : bytes;
  return new TextDecoder("utf-8").decode(raw);
}
