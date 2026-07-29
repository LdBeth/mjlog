// Disk-backed commentary drafts. The MCP server holds NO session state: every
// tool call carries the log reference, and the draft being built against a log
// lives here, under $HOME/.mjrender/drafts/<key>.json, keyed by the sha256 of
// the log's decoded XML text (computed by loadGameKeyed in core.ts) — the same
// game reaches the same draft whether it
// arrives as a gzipped .mjlog, a plain .xml, a renamed copy, or a tenhou.net
// URL. Writes are whole-file tmp+rename (atomic on one filesystem); two
// concurrent server processes on one log are last-write-wins by design.
//
// The on-disk shape is deliberately a valid weave input ({anchors, notes} —
// see weaveCommentary in core.ts), so a raw draft file can be fed straight to
// `cli.ts weave` with zero transformation.

import { dirname, join } from "node:path";
import type { StarNote, WeaveComment } from "./core.ts";

export interface Draft {
  version: 1;
  /** Last-seen source path/URL — informational only (the key is content-based). */
  log: string;
  savedAt: string;
  anchors: WeaveComment[];
  notes: StarNote[];
}

export function emptyDraft(log: string): Draft {
  return { version: 1, log, savedAt: "", anchors: [], notes: [] };
}

export function draftDir(): string {
  const home = Deno.env.get("HOME") ?? Deno.cwd();
  return join(home, ".mjrender", "drafts");
}

export function draftPath(key: string): string {
  return join(draftDir(), `${key}.json`);
}

/** Read the draft for a key; a missing file is an empty draft, not an error. */
export async function loadDraft(key: string, log: string): Promise<Draft> {
  try {
    const parsed = JSON.parse(await Deno.readTextFile(draftPath(key)));
    return {
      version: 1,
      log: typeof parsed.log === "string" ? parsed.log : log,
      savedAt: typeof parsed.savedAt === "string" ? parsed.savedAt : "",
      anchors: Array.isArray(parsed.anchors) ? parsed.anchors : [],
      notes: Array.isArray(parsed.notes) ? parsed.notes : [],
    };
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) return emptyDraft(log);
    throw e;
  }
}

/** Atomically replace the draft file for a key (mkdir -p, write tmp, rename). */
export async function saveDraft(key: string, draft: Draft): Promise<string> {
  const dest = draftPath(key);
  await Deno.mkdir(dirname(dest), { recursive: true });
  const tmp = `${dest}.tmp-${Deno.pid}`;
  await Deno.writeTextFile(tmp, JSON.stringify({ ...draft, savedAt: new Date().toISOString() }));
  await Deno.rename(tmp, dest);
  return dest;
}

/** Delete the draft for a key; false when no draft file exists. */
export async function deleteDraft(key: string): Promise<boolean> {
  try {
    await Deno.remove(draftPath(key));
    return true;
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) return false;
    throw e;
  }
}
