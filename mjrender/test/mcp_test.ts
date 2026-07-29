// End-to-end MCP test for the STATELESS server (v0.7.0): bundle src/mcp.ts
// (the same artifact `deno task bundle` ships), spawn the bundle over stdio,
// and drive a real JSON-RPC exchange. There is no session to establish and
// nothing is gated: every tool takes `log`, any round is readable/commentable
// at any time, and the draft lives on disk under $HOME/.mjrender/drafts — so
// each server process gets a throwaway HOME and the draft file itself is part
// of the assertions.
//
// Expected anchor ids and ★ sites are DERIVED at runtime from core.ts against
// the sample (robust to the sample's exact shape); only the wind layout (East
// rounds then South) is assumed from the derived winds.

import {
  listAnchors,
  listStarSites,
  loadGame,
  riichiDeclarations,
  roundLabel,
  uniqueRound,
  weaveCommentary,
} from "../src/core.ts";
import type { Beat } from "../src/model.ts";

const ROOT = new URL("../", import.meta.url).pathname;
const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;
// the SAME game, gzipped: drafts are keyed by the decoded XML's sha256, so
// both paths must reach one draft file.
const SAMPLE_GZ = new URL("../../1.mjlog", import.meta.url).pathname;

async function bundleServer(outFile: string): Promise<void> {
  const { success, stderr } = await new Deno.Command("deno", {
    args: ["bundle", "-o", outFile, "src/mcp.ts"],
    cwd: ROOT,
    stdout: "null",
    stderr: "piped",
  }).output();
  if (!success) {
    throw new Error(`deno bundle failed:\n${new TextDecoder().decode(stderr)}`);
  }
}

// deno-lint-ignore no-explicit-any
type Json = any;

class McpClient {
  #proc: Deno.ChildProcess;
  #writer: WritableStreamDefaultWriter<Uint8Array>;
  #lines: AsyncIterator<string>;
  #seq = 0;

  /** `home` isolates the spawned server's ~/.mjrender/drafts tree. */
  constructor(server: string, home: string) {
    this.#proc = new Deno.Command("deno", {
      args: ["run", "--allow-read", "--allow-write", "--allow-env=HOME", server],
      cwd: ROOT,
      env: { HOME: home },
      stdin: "piped",
      stdout: "piped",
      stderr: "null",
    }).spawn();
    this.#writer = this.#proc.stdin.getWriter();
    this.#lines = this.#proc.stdout
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TextLineStream())[Symbol.asyncIterator]();
  }

  async rpc(method: string, params: Json): Promise<Json> {
    const id = ++this.#seq;
    await this.send({ jsonrpc: "2.0", id, method, params });
    // read until the matching response id (skip notifications/log lines)
    while (true) {
      const { value, done } = await this.#lines.next();
      if (done) throw new Error("server closed stdout");
      if (!value.trim()) continue;
      const msg = JSON.parse(value);
      if (msg.id === id) {
        if (msg.error) throw new Error(`rpc error: ${JSON.stringify(msg.error)}`);
        return msg.result;
      }
    }
  }

  /** tools/call convenience — returns the raw ToolResult. */
  call(name: string, args: Json): Promise<Json> {
    return this.rpc("tools/call", { name, arguments: args });
  }

  async send(msg: Json): Promise<void> {
    await this.#writer.write(new TextEncoder().encode(JSON.stringify(msg) + "\n"));
  }

  async close(): Promise<void> {
    await this.#writer.close();
    await this.#proc.status;
  }
}

// minimal line splitter (avoid a std dep)
class TextLineStream extends TransformStream<string, string> {
  constructor() {
    let buf = "";
    super({
      transform(chunk, controller) {
        buf += chunk;
        const parts = buf.split("\n");
        buf = parts.pop() ?? "";
        for (const p of parts) controller.enqueue(p);
      },
      flush(controller) {
        if (buf) controller.enqueue(buf);
      },
    });
  }
}

const txt = (r: Json): string => r.content[0].text as string;
function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}
function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}

/**
 * initialize + serverInfo check + initialized notification, once per server.
 * The handshake deliberately speaks the LEGACY 2024-11-05 protocol version:
 * serveStdio must keep answering old clients.
 */
async function handshake(c: McpClient): Promise<void> {
  const init = await c.rpc("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "mjrender-test", version: "0.0.0" },
  });
  assert(
    init.serverInfo?.name === "mjrender",
    `unexpected serverInfo: ${JSON.stringify(init.serverInfo)}`,
  );
  await c.send({ jsonrpc: "2.0", method: "notifications/initialized" });
}

// ---- shared across tests: expected shape from the same core the server
// uses, and ONE server bundle (the slowest setup step) ----
const game = await loadGame(SAMPLE);
const anchors: Beat[] = listAnchors(game);
const stars = listStarSites(game);
const nRounds = game.rounds.length;
const anchorsOf = (r: number) => anchors.filter((b) => b.round === r);
const starsOf = (r: number) => stars.filter((s) => s.round === r);
const commentText = (id: number) => `テスト解説#${id}。`;

const bundleDir = await Deno.makeTempDir();
const server = `${bundleDir}/mcp.mjs`;
await bundleServer(server);
globalThis.addEventListener("unload", () => {
  try {
    Deno.removeSync(bundleDir, { recursive: true });
  } catch { /* best effort */ }
});

const TOOLS = [
  "mj_add_comment",
  "mj_add_note",
  "mj_clear_draft",
  "mj_draft_status",
  "mj_get_kyoku_result",
  "mj_get_kyoku_start",
  "mj_get_riichi_declarations",
  "mj_get_snapshot",
  "mj_list_anchors",
  "mj_outline",
  "mj_render_kyoku",
  "mj_weave_commentary",
];

Deno.test({
  name: "mcp: stateless one-shot flow (nothing gated, draft on disk)",
  // the server child process lives across steps by design
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const chukan = anchors.find((b) => b.kind === "中間総括");
    const owari = anchors.find((b) => b.kind === "終局総括");
    assert(chukan, "sample must have a 中間総括 anchor");
    assert(owari, "sample must have a 終局総括 anchor");
    // "E1" is the wind form of round 0 — the ★-note selector used below
    assert(roundLabel(game, 0).startsWith("東1局"), "round 0 must be 東1局 for the E1 selector");
    const lateRound = nRounds - 1;
    const s1Round = uniqueRound(game, "S1"); // a late round, readable straight away

    const home = await Deno.makeTempDir();
    const tmp = await Deno.makeTempDir();
    const c = new McpClient(server, home);
    try {
      await handshake(c);

      await t.step("tools/list: the 12 stateless tools, with annotations", async () => {
        const tools = await c.rpc("tools/list", {});
        const names = tools.tools.map((x: Json) => x.name).sort();
        assert(
          JSON.stringify(names) === JSON.stringify(TOOLS),
          `tool set mismatch: ${names}`,
        );
        const outline = tools.tools.find((x: Json) => x.name === "mj_outline");
        assert(
          outline?.annotations?.readOnlyHint === true,
          `mj_outline must advertise readOnlyHint: ${JSON.stringify(outline?.annotations)}`,
        );
        const clear = tools.tools.find((x: Json) => x.name === "mj_clear_draft");
        assert(
          clear?.annotations?.destructiveHint === true,
          `mj_clear_draft must advertise destructiveHint: ${JSON.stringify(clear?.annotations)}`,
        );
      });

      await t.step("a bad log reference errors (no session to blame)", async () => {
        const bad = await c.call("mj_list_anchors", { log: `${tmp}/does-not-exist.xml` });
        assert(bad.isError, `nonexistent log must error: ${txt(bad)}`);
      });

      await t.step("mj_outline: legend EVERY time, coverage, results visible", async () => {
        const first = await c.call("mj_outline", { log: SAMPLE });
        assert(!first.isError, `outline errored: ${txt(first)}`);
        assert(txt(first).includes("■この牌譜の読み方"), "outline must carry the legend");
        assert(txt(first).includes("draft: 0/"), `outline missing coverage: ${txt(first)}`);
        assert(txt(first).includes("◆終局"), "outline must show results (not a spoiler shield)");
        // stateless: the legend is idempotent, not once-per-process
        const second = await c.call("mj_outline", { log: SAMPLE });
        assert(
          txt(second).includes("■この牌譜の読み方"),
          "the legend must repeat on every mj_outline call",
        );
      });

      await t.step("ungated reads: any round, any anchor, all riichi", async () => {
        const late = await c.call("mj_render_kyoku", { log: SAMPLE, kyoku: "S1" });
        assert(!late.isError, `a late round must render immediately: ${txt(late)}`);
        for (const b of anchorsOf(s1Round)) {
          assert(txt(late).includes(`〔解説ポイント#${b.id}:`), `S1 render missing #${b.id}`);
        }
        const res = await c.call("mj_get_kyoku_result", { log: SAMPLE, kyoku: String(lateRound) });
        assert(!res.isError, `last round's result must be readable: ${txt(res)}`);
        assert(JSON.parse(txt(res)).length >= 1, `expected a result entry: ${txt(res)}`);
        const lateAnchor = anchorsOf(lateRound)[0];
        const snap = await c.call("mj_get_snapshot", { log: SAMPLE, anchor: lateAnchor.id });
        assert(!snap.isError, `a late anchor's snapshot must work: ${txt(snap)}`);
        for (const needle of ["┌盤面", "残り山", "手牌 P0:"]) {
          assert(txt(snap).includes(needle), `snapshot missing ${needle}: ${txt(snap)}`);
        }
        // no kyoku → the WHOLE game's declarations, with no "未開放" caveat
        const riichi = await c.call("mj_get_riichi_declarations", { log: SAMPLE });
        assert(!riichi.isError, `riichi list errored: ${txt(riichi)}`);
        assert(!txt(riichi).includes("未開放"), `riichi list must not be filtered: ${txt(riichi)}`);
        assert(
          JSON.parse(txt(riichi)).length === riichiDeclarations(game).length,
          `riichi count mismatch: ${txt(riichi)}`,
        );
      });

      await t.step("mj_render_kyoku 0: inline boards, no 結果時点 block, no legend", async () => {
        const k0 = await c.call("mj_render_kyoku", { log: SAMPLE, kyoku: "0" });
        assert(!k0.isError, `render 0 errored: ${txt(k0)}`);
        assert(txt(k0).includes("┌盤面"), "render must contain an inline board block");
        // the end-of-hand ground-truth block folds into the final snapshot in inline mode
        assert(
          !txt(k0).includes("◇結果時点の各家手牌:"),
          "inline render must omit the 結果時点 block",
        );
        // 配牌評価 (and 中間総括/終局総括) carry no inline snapshot — the deal block is the board
        const boards0 = txt(k0).split("┌盤面").length - 1;
        const wantBoards0 = anchorsOf(0)
          .filter((b) => !["配牌評価", "中間総括", "終局総括"].includes(b.kind)).length;
        assert(
          boards0 === wantBoards0,
          `render 0 has ${boards0} inline boards, want ${wantBoards0} (no snapshot at 配牌評価)`,
        );
        assert(!txt(k0).includes("■この牌譜の読み方"), "kyoku render must not carry the legend");
        for (const b of anchorsOf(0)) {
          assert(txt(k0).includes(`〔解説ポイント#${b.id}:`), `render 0 missing anchor #${b.id}`);
        }
      });

      await t.step("comments: any order, replace, atomic batches, 10-entry cap", async () => {
        // a LATE round first — commentary order is the model's business
        const lateId = anchorsOf(lateRound)[0].id;
        const lateSave = await c.call("mj_add_comment", {
          log: SAMPLE,
          comments: [{ anchor: lateId, text: commentText(lateId) }],
        });
        assert(!lateSave.isError, `late-round comment must save: ${txt(lateSave)}`);
        assert(txt(lateSave).includes(`saved #${lateId}`), `unexpected reply: ${txt(lateSave)}`);
        // ...then round 0
        const firstId = anchorsOf(0)[0].id;
        const early = await c.call("mj_add_comment", {
          log: SAMPLE,
          comments: [{ anchor: firstId, text: commentText(firstId) }],
        });
        assert(!early.isError, `round-0 comment must save: ${txt(early)}`);
        // re-saving replaces
        const again = await c.call("mj_add_comment", {
          log: SAMPLE,
          comments: [{ anchor: firstId, text: "改訂版。" }],
        });
        assert(
          !again.isError && txt(again).includes(`replaced #${firstId}`),
          `re-save should report a replace: ${txt(again)}`,
        );

        // atomicity: one bad entry saves NOTHING (the good id stays ・)
        const virginId = anchorsOf(3)[0].id;
        const atomic = await c.call("mj_add_comment", {
          log: SAMPLE,
          comments: [
            { anchor: virginId, text: "巻き添え。" },
            { anchor: 99999, text: "範囲外。" },
          ],
        });
        assert(atomic.isError, "a batch with a bad entry must fail");
        const st = await c.call("mj_draft_status", { log: SAMPLE });
        assert(
          txt(st).includes(`・ #${virginId}\t`),
          `atomic reject leaked #${virginId}: ${txt(st)}`,
        );
        assert(txt(st).includes("draft file: "), `status missing the draft path: ${txt(st)}`);

        // the max-10 cap is schema-level: an isError result OR a JSON-RPC error
        let capRejected = false;
        try {
          const big = Array.from({ length: 11 }, (_, k) => ({ anchor: k + 1, text: "多すぎ。" }));
          const r = await c.call("mj_add_comment", { log: SAMPLE, comments: big });
          capRejected = !!r.isError;
        } catch {
          capRejected = true; // schema violations may surface as JSON-RPC errors
        }
        assert(capRejected, "an 11-entry batch must be rejected");
      });

      await t.step("★ notes: kyoku-addressed save → replace → delete", async () => {
        const bad = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ kyoku: "E1", junme: 99, seat: 0, text: "場所なし。" }],
        });
        assert(bad.isError, "a non-★ position must error");
        assert(txt(bad).includes("★:"), `the error should list the ★ sites: ${txt(bad)}`);

        const s0 = starsOf(0)[0];
        const site = { kyoku: "E1", junme: s0.junme, seat: s0.seat };
        const dup = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ ...site, text: "一つ目。" }, { ...site, text: "二つ目。" }],
        });
        assert(
          dup.isError && txt(dup).includes("duplicate"),
          `duplicate ★ must error: ${txt(dup)}`,
        );

        const saved = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ ...site, text: "★一言。" }],
        });
        assert(!saved.isError && txt(saved).includes("saved 1"), `note save failed: ${txt(saved)}`);
        const replaced = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ ...site, text: "★一言（改）。" }],
        });
        assert(
          !replaced.isError && txt(replaced).includes("replaced"),
          `note re-save should replace: ${txt(replaced)}`,
        );
        const deleted = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ ...site, text: "  " }],
        });
        assert(
          !deleted.isError && txt(deleted).includes("deleted 1"),
          `blank text must delete: ${txt(deleted)}`,
        );
        const reDelete = await c.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ ...site, text: "" }],
        });
        assert(reDelete.isError, "deleting a never-saved note must error");
      });

      await t.step("fill every anchor (batches of 10), weaving on the way", async () => {
        let partialWeaveSeen = false;
        const groups = chunk(anchors.map((b) => b.id), 10);
        for (const [i, grp] of groups.entries()) {
          const r = await c.call("mj_add_comment", {
            log: SAMPLE,
            comments: grp.map((id) => ({ anchor: id, text: commentText(id) })),
          });
          assert(!r.isError, `fill ${grp} failed: ${txt(r)}`);
          if (i === 1) {
            // mid-fill: the weave still runs, but warns loudly
            const pw = await c.call("mj_weave_commentary", {
              log: SAMPLE,
              out: `${tmp}/partial.txt`,
            });
            assert(!pw.isError, `partial weave errored: ${txt(pw)}`);
            assert(
              txt(pw).split("\n")[0].startsWith("warning: partial weave"),
              `partial weave should warn on line 1: ${txt(pw)}`,
            );
            partialWeaveSeen = true;
          }
          if (i === groups.length - 1) {
            assert(
              txt(r).includes("all anchors filled"),
              `the last batch should report completion: ${txt(r)}`,
            );
          }
        }
        assert(partialWeaveSeen, "the mid-fill partial weave never ran");

        const finalOut = `${tmp}/woven.txt`;
        const wove = await c.call("mj_weave_commentary", { log: SAMPLE, out: finalOut });
        const summary = txt(wove);
        assert(!wove.isError, `final weave errored: ${summary}`);
        assert(summary.includes(finalOut), `weave summary missing path: ${summary}`);
        assert(!summary.includes("warning: partial weave"), `full weave must not warn: ${summary}`);

        const doc = await Deno.readTextFile(finalOut);
        assert(doc.includes("== 南入 =="), "woven doc missing the 南入 interlude");
        assert(
          doc.includes("◆解説（中間総括）:"),
          "woven doc missing the 中間総括 commentary line",
        );
        assert(doc.includes(commentText(chukan.id)), "woven doc missing the spliced 中間総括 text");
        // no placeholder LINE survives (the reader preamble legitimately
        // explains the 〔解説ポイント#N: …〕 notation, hence the \d+ anchor)
        const leftover = doc.split("\n").filter((l) => /^〔解説ポイント#\d+:/.test(l));
        assert(
          leftover.length === 0,
          `anchor placeholders survived a complete fill: ${leftover.join(" / ")}`,
        );
      });

      await t.step("mj_clear_draft wipes the disk draft", async () => {
        const cleared = await c.call("mj_clear_draft", { log: SAMPLE });
        assert(!cleared.isError, `clear errored: ${txt(cleared)}`);
        assert(txt(cleared).startsWith("deleted "), `unexpected clear reply: ${txt(cleared)}`);
        const st = await c.call("mj_draft_status", { log: SAMPLE });
        assert(txt(st).includes("draft: 0/"), `draft should be empty after clear: ${txt(st)}`);
      });
    } finally {
      await c.close();
      await Deno.remove(tmp, { recursive: true }).catch(() => {});
      await Deno.remove(home, { recursive: true }).catch(() => {});
    }
  },
});

// The stateless recovery story: there is no restore call. Server A saves
// commentary (every add is an atomic disk write), then "crashes"; a fresh
// Server B on the same HOME sees the identical draft simply by being handed
// the same log. The draft file is also asserted to be a valid weave input
// with zero transformation.
Deno.test({
  name: "mcp: drafts persist across restarts (no restore step)",
  sanitizeOps: false,
  sanitizeResources: false,
  async fn(t) {
    const noteText = "リーチ★メモ。";
    const r01 = anchors.filter((b) => b.round <= 1);
    const s0 = starsOf(0)[0];
    assert(s0, "sample must have a round-0 ★ site");
    const round0Anchor = anchorsOf(0)[0].id;

    const home = await Deno.makeTempDir(); // shared by both server processes
    const tmp = await Deno.makeTempDir();
    const fileA = `${tmp}/woven_A.txt`;
    const fileB = `${tmp}/woven_B.txt`;

    const A = new McpClient(server, home);
    let B: McpClient | undefined;
    try {
      // ================= Server A: the pre-crash session =================
      await handshake(A);
      await t.step("server A: comment rounds 0-1, note, partial weave", async () => {
        for (const r of [0, 1]) {
          const rk = await A.call("mj_render_kyoku", { log: SAMPLE, kyoku: String(r) });
          assert(!rk.isError, `A render ${r}: ${txt(rk)}`);
          const saved = await A.call("mj_add_comment", {
            log: SAMPLE,
            comments: anchorsOf(r).map((b) => ({ anchor: b.id, text: commentText(b.id) })),
          });
          assert(!saved.isError, `A fill ${r}: ${txt(saved)}`);
        }
        const note = await A.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ kyoku: "E1", junme: s0.junme, seat: s0.seat, text: noteText }],
        });
        assert(!note.isError && txt(note).includes("saved 1"), `A note: ${txt(note)}`);

        const wove = await A.call("mj_weave_commentary", { log: SAMPLE, out: fileA });
        assert(!wove.isError, `A weave: ${txt(wove)}`);
        assert(
          txt(wove).includes("warning: partial weave"),
          `A weave should warn (rounds 2+ unfilled): ${txt(wove)}`,
        );
      });
      await A.close(); // ---- simulated crash: nothing is flushed on exit ----

      await t.step("the draft file on disk is a valid weave input as-is", async () => {
        const files: string[] = [];
        for await (const e of Deno.readDir(`${home}/.mjrender/drafts`)) {
          if (e.isFile && e.name.endsWith(".json")) files.push(e.name);
        }
        assert(files.length === 1, `expected exactly one draft file, got ${files.join(" ")}`);
        const parsed = JSON.parse(
          await Deno.readTextFile(`${home}/.mjrender/drafts/${files[0]}`),
        );
        assert(parsed.version === 1, `unexpected draft version: ${parsed.version}`);
        assert(
          JSON.stringify(
            [...parsed.anchors].map((a: Json) => a.anchor).sort((x: number, y: number) => x - y),
          ) ===
            JSON.stringify(r01.map((b) => b.id)),
          `draft anchors mismatch: ${JSON.stringify(parsed.anchors)}`,
        );
        assert(
          parsed.notes.length === 1,
          `draft should hold 1 note: ${JSON.stringify(parsed.notes)}`,
        );
        assert(parsed.notes[0].text === noteText, "the saved ★ text must round-trip");
        // fed straight to the core weaver, the file reproduces Server A's document
        const r = weaveCommentary(game, parsed, {});
        assert(
          r.text + "\n" === await Deno.readTextFile(fileA),
          "the disk draft must weave byte-identically to the server's own output",
        );
      });

      // ================= Server B: a fresh process, same HOME =================
      B = new McpClient(server, home);
      await handshake(B);
      await t.step("server B: same draft, no restore call", async () => {
        const st = await B!.call("mj_draft_status", { log: SAMPLE });
        assert(!st.isError, `B status errored: ${txt(st)}`);
        const want = `draft: ${r01.length}/${anchors.length} comments, 1 notes`;
        assert(txt(st).includes(want), `B lost the draft (want "${want}"): ${txt(st)}`);

        const wove = await B!.call("mj_weave_commentary", { log: SAMPLE, out: fileB });
        assert(!wove.isError, `B weave: ${txt(wove)}`);
        assert(
          await Deno.readTextFile(fileA) === await Deno.readTextFile(fileB),
          "a restarted server must weave byte-identically to the pre-crash one",
        );
      });

      await t.step("server B: past rounds stay writable (no gate, no window)", async () => {
        const rev = await B!.call("mj_add_comment", {
          log: SAMPLE,
          comments: [{ anchor: round0Anchor, text: "改訂版。" }],
        });
        assert(
          !rev.isError && txt(rev).includes("replaced"),
          `revising a round-0 anchor should replace: ${txt(rev)}`,
        );
        const s0b = starsOf(0)[1] ?? s0;
        const note = await B!.call("mj_add_note", {
          log: SAMPLE,
          notes: [{ kyoku: "E1", junme: s0b.junme, seat: s0b.seat, text: "後付けメモ。" }],
        });
        assert(
          !note.isError && txt(note).includes("saved"),
          `a round-0 ★ note must still be addressable: ${txt(note)}`,
        );
      });

      await t.step("the draft is content-keyed: .xml and .mjlog share it", async () => {
        const xml = await B!.call("mj_draft_status", { log: SAMPLE });
        const gz = await B!.call("mj_draft_status", { log: SAMPLE_GZ });
        assert(!gz.isError, `gz status errored: ${txt(gz)}`);
        const coverage = (s: string) => s.split("\n")[0].split(" — ")[1];
        assert(
          coverage(txt(xml)) === coverage(txt(gz)),
          `same game, different container: "${coverage(txt(xml))}" vs "${coverage(txt(gz))}"`,
        );
        // ...and literally the same file
        const path = (s: string) => s.split("\n")[1];
        assert(path(txt(xml)) === path(txt(gz)), `draft paths diverge: ${path(txt(gz))}`);
      });
    } finally {
      await A.close().catch(() => {}); // double-close after the simulated crash is fine
      if (B) await B.close().catch(() => {});
      await Deno.remove(tmp, { recursive: true }).catch(() => {});
      await Deno.remove(home, { recursive: true }).catch(() => {});
    }
  },
});
