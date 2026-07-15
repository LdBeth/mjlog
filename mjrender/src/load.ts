// Read a mjlog file and return its XML text, transparently decompressing gzip.

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

export async function loadXml(path: string): Promise<string> {
  const bytes = await Deno.readFile(path);
  const raw = isGzip(bytes) ? await gunzip(bytes) : bytes;
  return new TextDecoder("utf-8").decode(raw);
}
