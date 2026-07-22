// End-to-end MCP test for the PACED, kyoku-gated flow (v0.6.0): bundle
// src/mcp.ts (the same artifact `deno task bundle` ships), spawn the bundle
// over stdio, and drive a real JSON-RPC exchange that walks the whole gated
// loop — open → orient → per-kyoku render/comment/advance → weave.
//
// Expected per-round anchor ids and ★ sites are DERIVED at runtime from
// core.ts against the sample (robust to the sample's exact shape); only the
// wind layout (East rounds then South) is assumed from the derived winds.

import { listAnchors, listStarSites, loadGame } from "../src/core.ts";
import type { Beat } from "../src/model.ts";

const ROOT = new URL("../", import.meta.url).pathname;
const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

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

  constructor(server: string) {
    this.#proc = new Deno.Command("deno", {
      args: ["run", "--allow-read", "--allow-write", "--allow-env=HOME", server],
      cwd: ROOT,
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

/** initialize + serverInfo check + initialized notification, once per server. */
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
const commentText = (id: number) => `テスト解説#${id}。`;

const bundleDir = await Deno.makeTempDir();
const server = `${bundleDir}/mcp.mjs`;
await bundleServer(server);
globalThis.addEventListener("unload", () => {
  try {
    Deno.removeSync(bundleDir, { recursive: true });
  } catch { /* best effort */ }
});

Deno.test("mcp: paced kyoku-gated commentary flow end-to-end", async () => {
  const filledBefore = (r: number) => anchors.filter((b) => b.round < r).length;
  const unlockedAt = (r: number) => anchors.filter((b) => b.round <= r).length;
  const winds = game.rounds.map((rd) => rd.kyoku >> 2);
  const chukan = anchors.find((b) => b.kind === "中間総括");
  const owari = anchors.find((b) => b.kind === "終局総括");
  assert(chukan, "sample must have a 中間総括 anchor");
  assert(owari, "sample must have a 終局総括 anchor");
  // The wind crossing round (last East round): winds[r] !== winds[r+1].
  const crossRound = winds.findIndex((w, i) => i + 1 < nRounds && winds[i + 1] !== w);
  assert(crossRound >= 0, "sample must cross a wind boundary");
  assert(chukan.round === crossRound, "中間総括 must sit on the wind-crossing round");

  const c = new McpClient(server);
  const tmp = await Deno.makeTempDir();
  try {
    await handshake(c);

    // ---- 1. tools/list: + mj_next_kyoku, − mj_get_final_standings ----
    const tools = await c.rpc("tools/list", {});
    const names = tools.tools.map((t: Json) => t.name).sort();
    const want = [
      "mj_add_comment",
      "mj_add_note",
      "mj_draft_status",
      "mj_get_kyoku_result",
      "mj_get_kyoku_start",
      "mj_get_riichi_declarations",
      "mj_get_snapshot",
      "mj_list_anchors",
      "mj_next_kyoku",
      "mj_open_log",
      "mj_render_game",
      "mj_render_kyoku",
      "mj_restore_state",
      "mj_weave_commentary",
    ];
    assert(JSON.stringify(names) === JSON.stringify(want), `tool set mismatch: ${names}`);

    // everything but mj_open_log errors before a log is open
    const early = await c.call("mj_list_anchors", {});
    assert(
      early.isError && txt(early).includes("mj_open_log"),
      `expected 'no log loaded': ${txt(early)}`,
    );

    // ---- 2. open: legend once, focus 東1, anchor count; reopen keeps, no legend ----
    const opened = await c.call("mj_open_log", { path: SAMPLE });
    assert(!opened.isError, `open failed: ${txt(opened)}`);
    assert(txt(opened).includes("■この牌譜の読み方"), "first open must carry the legend");
    assert(txt(opened).includes("focus: 東1局"), `open reply missing focus: ${txt(opened)}`);
    assert(
      txt(opened).includes(`anchors: ${anchors.length}`),
      `open reply missing anchor count: ${txt(opened)}`,
    );

    // a second (immediate) open keeps focus AND does NOT repeat the legend
    const reopen0 = await c.call("mj_open_log", { path: SAMPLE });
    assert(!txt(reopen0).includes("■この牌譜の読み方"), "legend must not repeat on reopen");
    assert(txt(reopen0).includes("focus: 東1局"), `reopen lost focus: ${txt(reopen0)}`);

    // ---- 3. mj_render_game at focus 0 is UNGATED (results visible) ----
    const outline = await c.call("mj_render_game", {});
    assert(!outline.isError, `outline errored: ${txt(outline)}`);
    assert(
      txt(outline).includes("【南1局") && txt(outline).includes("◆終局"),
      `outline missing South/終局: ${txt(outline).slice(0, 200)}`,
    );

    // ---- 4. gate errors for future kyoku (renders/snapshots/facts/writes) ----
    const gRender = await c.call("mj_render_kyoku", { kyoku: "S1" });
    assert(
      gRender.isError && txt(gRender).includes("locked"),
      `render S1 should lock: ${txt(gRender)}`,
    );
    const gSnapK = await c.call("mj_get_snapshot", { kyoku: "5", junme: 1 });
    assert(
      gSnapK.isError && txt(gSnapK).includes("locked"),
      `snapshot k5 should lock: ${txt(gSnapK)}`,
    );
    // future anchor id (a South-round anchor) locks the snapshot too
    const southAnchor = anchors.find((b) => winds[b.round] !== winds[0])!;
    const gSnapA = await c.call("mj_get_snapshot", { anchor: southAnchor.id });
    assert(
      gSnapA.isError && txt(gSnapA).includes("locked"),
      `snapshot future anchor should lock: ${txt(gSnapA)}`,
    );
    const futureRoundSel = String(southAnchor.round);
    const gResult = await c.call("mj_get_kyoku_result", { kyoku: futureRoundSel });
    assert(
      gResult.isError && txt(gResult).includes("locked"),
      `result future should lock: ${txt(gResult)}`,
    );
    const gComment = await c.call("mj_add_comment", {
      comments: [{ anchor: southAnchor.id, text: "早すぎ。" }],
    });
    assert(
      gComment.isError && txt(gComment).includes("locked"),
      `comment future should lock: ${txt(gComment)}`,
    );
    // (mj_add_note takes no kyoku argument — future rounds are unaddressable by design)
    // riichi without kyoku: only rounds <= focus (0) + the 未開放 note
    const riichi0 = await c.call("mj_get_riichi_declarations", {});
    assert(
      !riichi0.isError && txt(riichi0).includes("未開放局は含まず"),
      `riichi note missing: ${txt(riichi0)}`,
    );
    const shown0 = JSON.parse(txt(riichi0).split("\n（未開放")[0]);
    assert(shown0.every((d: Json) => d.roundIndex <= 0), "riichi list must be filtered to focus");

    // ---- 5. mj_render_kyoku "0": inline board block, no legend, anchor lines ----
    const k0 = await c.call("mj_render_kyoku", { kyoku: "0" });
    assert(!k0.isError, `render 0 errored: ${txt(k0)}`);
    assert(txt(k0).includes("┌盤面"), "render must contain an inline board block");
    // the end-of-hand ground-truth block folds into the final snapshot in inline mode
    assert(!txt(k0).includes("◇結果時点の各家手牌:"), "inline render must omit the 結果時点 block");
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

    // ---- 9a. edge cases at focus 0: batch atomicity, 11-cap, bad ★ ----
    const round0Ids = anchorsOf(0).map((b) => b.id);
    const atomic = await c.call("mj_add_comment", {
      comments: [{ anchor: round0Ids[0], text: "巻き添え。" }, { anchor: 99999, text: "范囲外。" }],
    });
    assert(atomic.isError, "atomic batch with a bad entry must fail");
    const st0 = await c.call("mj_draft_status", {});
    assert(
      txt(st0).includes(`・ #${round0Ids[0]}`),
      `atomic reject leaked #${round0Ids[0]}: ${txt(st0)}`,
    );

    let capRejected = false;
    try {
      const big = Array.from({ length: 11 }, (_, k) => ({ anchor: k + 1, text: "多すぎ。" }));
      const r = await c.call("mj_add_comment", { comments: big });
      capRejected = !!r.isError;
    } catch {
      capRejected = true; // schema violations may surface as JSON-RPC errors
    }
    assert(capRejected, "an 11-entry batch must be rejected");

    const badStar = await c.call("mj_add_note", {
      notes: [{ junme: 99, seat: 0, text: "場所なし。" }],
    });
    assert(badStar.isError, "a non-★ position must error");

    // ---- 6. mj_next_kyoku with round 0 unfilled: error listing the #ids ----
    const stuck = await c.call("mj_next_kyoku", {});
    assert(stuck.isError, "advance with unfilled anchors must error");
    for (const b of anchorsOf(0)) {
      assert(txt(stuck).includes(`#${b.id}(`), `unfilled error missing #${b.id}: ${txt(stuck)}`);
    }

    // ---- 7. main loop rounds 0..9 ----
    const fill = async (ids: number[]): Promise<Json> => {
      let last: Json;
      for (const grp of chunk(ids, 10)) {
        last = await c.call("mj_add_comment", {
          comments: grp.map((id) => ({ anchor: id, text: commentText(id) })),
        });
        assert(!last.isError, `fill ${grp} failed: ${txt(last)}`);
      }
      return last!;
    };

    let partialWeaveSeen = false;
    for (let r = 0; r < nRounds; r++) {
      // --- pre-fill checks (focus = r, round r not yet filled) ---
      if (r === 1) {
        // after mj_next_kyoku the note window STILL addresses the finished round 0
        // (it moves only when the new focus is rendered) — the advance HINT is actionable
        const s0 = stars.find((s) => s.round === 0)!;
        const graceNote = await c.call("mj_add_note", {
          notes: [{ junme: s0.junme, seat: s0.seat, text: "後から見ると遅い。" }],
        });
        assert(
          !graceNote.isError && txt(graceNote).includes("saved 1"),
          `grace-window note failed: ${txt(graceNote)}`,
        );
        const graceDel = await c.call("mj_add_note", {
          notes: [{ junme: s0.junme, seat: s0.seat, text: "" }],
        });
        assert(
          !graceDel.isError && txt(graceDel).includes("deleted 1"),
          `grace-window delete failed: ${txt(graceDel)}`,
        );
        // rendering the new focus moves the window to round 1 and locks round 0
        const k1 = await c.call("mj_render_kyoku", { kyoku: "1" });
        assert(!k1.isError, `render 1 errored: ${txt(k1)}`);
        const sites1 = stars.filter((s) => s.round === 1);
        const s0only = stars.find((s) =>
          s.round === 0 && !sites1.some((x) => x.junme === s.junme && x.seat === s.seat)
        );
        if (s0only) {
          const lockedNote = await c.call("mj_add_note", {
            notes: [{ junme: s0only.junme, seat: s0only.seat, text: "手遅れ。" }],
          });
          assert(
            lockedNote.isError && txt(lockedNote).includes("notes address"),
            `round-0 note must lock after rendering round 1: ${txt(lockedNote)}`,
          );
        }
        // a REVISION of an already-filled past anchor succeeds (replace-only)
        const past = anchorsOf(0)[0].id;
        const rev = await c.call("mj_add_comment", {
          comments: [{ anchor: past, text: "改訂版。" }],
        });
        assert(
          !rev.isError && txt(rev).includes("replaced"),
          `past revision should replace: ${txt(rev)}`,
        );
        // a ★ note in the FOCUS round (window = round 1): save → empty-text delete round-trips
        const s1 = sites1[0]!;
        const site = { junme: s1.junme, seat: s1.seat };
        const saved = await c.call("mj_add_note", { notes: [{ ...site, text: "リーチ一言。" }] });
        assert(!saved.isError && txt(saved).includes("saved 1"), `note save failed: ${txt(saved)}`);
        const deleted = await c.call("mj_add_note", { notes: [{ ...site, text: "  " }] });
        assert(
          !deleted.isError && txt(deleted).includes("deleted 1"),
          `note delete failed: ${txt(deleted)}`,
        );
        const reDelete = await c.call("mj_add_note", { notes: [{ ...site, text: "" }] });
        assert(reDelete.isError, "deleting a never-saved note must error");
        // a snapshot at an unlocked riichi anchor still recalls the board
        const riichiHere = anchors.find((b) => b.kind === "リーチ判断" && b.round <= 1);
        if (riichiHere) {
          const snap = await c.call("mj_get_snapshot", { anchor: riichiHere.id });
          for (const needle of ["┌盤面", "残り山", "手牌 P0:"]) {
            assert(txt(snap).includes(needle), `snapshot missing ${needle}: ${txt(snap)}`);
          }
        }
      }
      if (r === 2) {
        // reopen keeps the accumulated draft AND the advanced focus
        const reopened = await c.call("mj_open_log", { path: SAMPLE });
        assert(!txt(reopened).includes("■この牌譜の読み方"), "reopen must not repeat legend");
        const wantLine = `${filledBefore(2)}/${
          unlockedAt(2)
        } comments (kyoku 3/${nRounds} unlocked)`;
        assert(
          txt(reopened).includes(wantLine),
          `reopen lost draft/focus (want "${wantLine}"): ${txt(reopened)}`,
        );
      }
      if (r === nRounds - 1) {
        // at the final focus the render carries ◆終局 and the 終局総括 anchor
        const kLast = await c.call("mj_render_kyoku", { kyoku: String(r) });
        assert(!kLast.isError, `render last errored: ${txt(kLast)}`);
        assert(txt(kLast).includes("◆終局"), "final render missing ◆終局");
        assert(
          txt(kLast).includes(`〔解説ポイント#${owari.id}:`),
          "final render missing 終局総括 anchor",
        );
      }

      // --- fill the focus round's anchors, then advance ---
      const ids = anchorsOf(r).map((b) => b.id);
      if (r === crossRound) {
        // fill every regular anchor but withhold 中間総括 → checkpoint gate
        const regular = ids.filter((id) => id !== chukan.id);
        await fill(regular);
        const gate = await c.call("mj_next_kyoku", {});
        assert(gate.isError, "advance without 中間総括 must error");
        assert(
          txt(gate).includes("中間総括") && txt(gate).includes("点況"),
          `checkpoint missing 中間総括/点況: ${txt(gate)}`,
        );
        await fill([chukan.id]);
      } else {
        await fill(ids);
      }

      // --- post-fill check: a partial weave somewhere mid-loop warns loudly ---
      if (r === crossRound + 1 && !partialWeaveSeen) {
        const partialOut = `${tmp}/partial.txt`;
        const pw = await c.call("mj_weave_commentary", { out: partialOut });
        assert(
          !pw.isError && txt(pw).includes("warning: partial weave"),
          `partial weave should warn: ${txt(pw)}`,
        );
        partialWeaveSeen = true;
      }

      const adv = await c.call("mj_next_kyoku", {});
      if (r === 0) {
        assert(!adv.isError, `round-0 advance errored: ${txt(adv)}`);
        assert(txt(adv).includes("advanced:"), `round-0 advance missing 'advanced:': ${txt(adv)}`);
        assert(txt(adv).includes("STOP"), "round-0 advance missing STOP");
        assert(txt(adv).includes("HINT"), "round-0 advance missing HINT (no ★ notes saved yet)");
      }
      if (r === crossRound) {
        assert(!adv.isError, `wind-cross advance errored: ${txt(adv)}`);
        assert(txt(adv).includes("== 南入 =="), `wind crossing missing 南入: ${txt(adv)}`);
      }
      if (r === nRounds - 1) {
        // final kyoku: no advance — points to the weave + STOP
        assert(!adv.isError, `final next_kyoku errored: ${txt(adv)}`);
        assert(
          txt(adv).includes("mj_weave_commentary") && txt(adv).includes("STOP"),
          `final next_kyoku wording: ${txt(adv)}`,
        );
      }
    }
    assert(partialWeaveSeen, "the mid-loop partial weave never ran");

    // ---- 8. final weave: no partial warning; doc has the interlude + spliced comment ----
    const finalOut = `${tmp}/woven.txt`;
    const wove = await c.call("mj_weave_commentary", { out: finalOut });
    const summary = txt(wove);
    assert(!wove.isError, `final weave errored: ${summary}`);
    assert(summary.includes(finalOut), `weave summary missing path: ${summary}`);
    assert(!summary.includes("warning: partial weave"), `full weave must not warn: ${summary}`);
    const doc = await Deno.readTextFile(finalOut);
    assert(doc.includes("== 南入 =="), "woven doc missing the 南入 interlude");
    assert(doc.includes("◆解説（中間総括）:"), "woven doc missing the 中間総括 commentary line");
    assert(doc.includes(commentText(chukan.id)), "woven doc missing the spliced 中間総括 text");
    assert(
      !doc.includes(`〔解説ポイント#${chukan.id}:`),
      "filled anchor placeholder should be gone",
    );
  } finally {
    await c.close();
    await Deno.remove(tmp, { recursive: true });
  }
});

// mj_restore_state: after a SERVER RESTART the client re-sends its whole
// accumulated draft + focus in ONE bulk call. Server A plays rounds 0–1,
// advances to focus 2, and weaves; it then "crashes" (client closes). A fresh
// Server B reopens the log, bulk-restores from Server A's payload, and must
// weave a byte-identical document — plus honor the gate, replace-only, note
// window, atomicity, and past-gap warnings the same as a never-restarted run.
Deno.test("mcp: restart restore round-trips the draft", async () => {
  const noteText = "リーチ★メモ。";

  // the draft Server A accumulates: every anchor in rounds 0..1, plus one ★
  // note at a round-0 site. Server B must reproduce it exactly.
  const r01 = anchors.filter((b) => b.round <= 1);
  const unlocked2 = anchors.filter((b) => b.round <= 2).length;
  const s0 = stars.find((s) => s.round === 0);
  assert(s0, "sample must have a round-0 ★ site");
  // a round-0 site that is NOT also a ★ site in the focus round (round 2), so a
  // note addressed to it genuinely tests the moved note window rather than
  // coincidentally landing on a same-(junme,seat) focus-round site.
  const s0Only = stars.find((s) =>
    s.round === 0 && !stars.some((x) => x.round === 2 && x.junme === s.junme && x.seat === s.seat)
  );
  assert(anchorsOf(3).length > 0, "sample must have a round-3 anchor (locked beyond focus 2)");
  const round0Anchor = anchorsOf(0)[0].id;
  const round3Anchor = anchorsOf(3)[0].id;

  const tmp = await Deno.makeTempDir();
  const fileA = `${tmp}/woven_A.txt`;
  const fileB = `${tmp}/woven_B.txt`;

  const restoreComments = (ids: number[]) =>
    ids.map((id) => ({ anchor: id, text: commentText(id) }));

  const A = new McpClient(server);
  let B: McpClient | undefined;
  try {
    // ================= Server A: the pre-crash session =================
    await handshake(A);

    const openedA = await A.call("mj_open_log", { path: SAMPLE });
    assert(!openedA.isError, `A open failed: ${txt(openedA)}`);

    // round 0: render → fill anchors → one ★ note at a round-0 site
    const rkA0 = await A.call("mj_render_kyoku", { kyoku: "0" });
    assert(!rkA0.isError, `A render 0: ${txt(rkA0)}`);
    const cA0 = await A.call("mj_add_comment", {
      comments: restoreComments(anchorsOf(0).map((b) => b.id)),
    });
    assert(!cA0.isError, `A fill 0: ${txt(cA0)}`);
    const nA = await A.call("mj_add_note", {
      notes: [{ junme: s0.junme, seat: s0.seat, text: noteText }],
    });
    assert(!nA.isError && txt(nA).includes("saved 1"), `A note: ${txt(nA)}`);

    const advA0 = await A.call("mj_next_kyoku", {});
    assert(!advA0.isError, `A advance 0: ${txt(advA0)}`);

    // round 1: render → fill anchors
    const rkA1 = await A.call("mj_render_kyoku", { kyoku: "1" });
    assert(!rkA1.isError, `A render 1: ${txt(rkA1)}`);
    const cA1 = await A.call("mj_add_comment", {
      comments: restoreComments(anchorsOf(1).map((b) => b.id)),
    });
    assert(!cA1.isError, `A fill 1: ${txt(cA1)}`);

    const advA1 = await A.call("mj_next_kyoku", {});
    assert(!advA1.isError, `A advance 1: ${txt(advA1)}`); // focus now round 2

    // partial weave (rounds 2..9 unfilled) → warns
    const woveA = await A.call("mj_weave_commentary", { out: fileA });
    assert(!woveA.isError, `A weave: ${txt(woveA)}`);
    assert(
      txt(woveA).includes("warning: partial weave"),
      `A weave should warn partial (round 2 unfilled): ${txt(woveA)}`,
    );

    await A.close(); // ---- simulated crash ----

    // ================= Server B: fresh process =================
    B = new McpClient(server);
    await handshake(B);

    // (a) restore before any open → error mentioning mj_open_log
    const preOpen = await B.call("mj_restore_state", {
      focus: "2",
      comments: restoreComments(r01.map((b) => b.id)),
    });
    assert(
      preOpen.isError && txt(preOpen).includes("mj_open_log"),
      `restore before open should demand mj_open_log: ${txt(preOpen)}`,
    );

    // (b) open → focus round 0, empty draft
    const openedB = await B.call("mj_open_log", { path: SAMPLE });
    assert(!openedB.isError, `B open failed: ${txt(openedB)}`);
    assert(txt(openedB).includes("focus: 東1局"), `B open lost focus: ${txt(openedB)}`);

    // (c) rejection + atomicity: focus "2" but a round-3 comment is beyond focus →
    //     error, and (atomic) the accompanying round-0 comment must NOT leak.
    const bad = await B.call("mj_restore_state", {
      focus: "2",
      comments: [
        { anchor: round0Anchor, text: commentText(round0Anchor) },
        { anchor: round3Anchor, text: "越境。" },
      ],
    });
    assert(
      bad.isError && txt(bad).includes("beyond the restored focus"),
      `restore with a beyond-focus anchor must error: ${txt(bad)}`,
    );
    const st = await B.call("mj_draft_status", {});
    assert(
      txt(st).includes("draft: 0/") && txt(st).includes("0 notes"),
      `draft must stay empty after an atomic reject: ${txt(st)}`,
    );

    // (d) real restore: focus "2", ALL round-0+1 comments in ONE call + the ★
    //     note with a wind-form selector ("E1") to exercise uniqueRound norm.
    const restored = await B.call("mj_restore_state", {
      focus: "2",
      comments: restoreComments(r01.map((b) => b.id)),
      notes: [{ kyoku: "E1", junme: s0.junme, seat: s0.seat, text: noteText }],
    });
    assert(!restored.isError, `restore failed: ${txt(restored)}`);
    assert(txt(restored).includes("restored:"), `restore reply missing 'restored:': ${txt(restored)}`);
    const wantDraft =
      `draft: ${r01.length}/${unlocked2} comments (kyoku 3/${nRounds} unlocked), 1 notes`;
    assert(
      txt(restored).includes(wantDraft),
      `restore draft line (want "${wantDraft}"): ${txt(restored)}`,
    );
    assert(
      !txt(restored).includes("warning:"),
      `restore with no past gaps must not warn: ${txt(restored)}`,
    );

    // (e) gating intact: render round 3 → locked; mj_next_kyoku → lists round-2 ids
    const gRender = await B.call("mj_render_kyoku", { kyoku: "3" });
    assert(
      gRender.isError && txt(gRender).includes("locked"),
      `render round 3 should lock beyond focus 2: ${txt(gRender)}`,
    );
    const gAdv = await B.call("mj_next_kyoku", {});
    assert(gAdv.isError, "advance with round-2 anchors unfilled must error");
    for (const b of anchorsOf(2)) {
      assert(txt(gAdv).includes(`#${b.id}(`), `advance error missing #${b.id}: ${txt(gAdv)}`);
    }

    // (h) weave equality — the restored draft still matches Server A byte-for-byte.
    //     Done BEFORE the mutating steps (f/g) so the equality holds.
    const woveB = await B.call("mj_weave_commentary", { out: fileB });
    assert(!woveB.isError, `B weave: ${txt(woveB)}`);
    const docA = await Deno.readTextFile(fileA);
    const docB = await Deno.readTextFile(fileB);
    assert(docA === docB, "a restored session must weave byte-identically to the never-restarted one");

    // (f) noteRound = focus: a round-0-only ★ note now locks; a round-2 ★ note saves.
    if (s0Only) {
      const noteRound0 = await B.call("mj_add_note", {
        notes: [{ junme: s0Only.junme, seat: s0Only.seat, text: "手遅れ。" }],
      });
      assert(
        noteRound0.isError && txt(noteRound0).includes("notes address"),
        `a round-0 ★ note must lock after restore to focus 2: ${txt(noteRound0)}`,
      );
    }
    const s2 = stars.find((s) => s.round === 2);
    if (s2) {
      const noteRound2 = await B.call("mj_add_note", {
        notes: [{ junme: s2.junme, seat: s2.seat, text: "焦点局メモ。" }],
      });
      assert(
        !noteRound2.isError && txt(noteRound2).includes("saved 1"),
        `a focus-round (round 2) ★ note should save: ${txt(noteRound2)}`,
      );
    }

    // (g) replace-only preserved: revising a restored round-0 anchor succeeds.
    const rev = await B.call("mj_add_comment", {
      comments: [{ anchor: round0Anchor, text: "改訂版。" }],
    });
    assert(
      !rev.isError && txt(rev).includes("replaced"),
      `revising a restored past anchor should replace: ${txt(rev)}`,
    );

    // (i) gap warning + idempotency: restore omitting a round-0 anchor warns; a
    //     fresh mj_add_comment fill of that gap is replace-only-rejected; a full
    //     restore then comes back clean.
    const gapRestore = await B.call("mj_restore_state", {
      focus: "2",
      comments: restoreComments(r01.filter((b) => b.id !== round0Anchor).map((b) => b.id)),
    });
    assert(
      !gapRestore.isError && txt(gapRestore).includes("warning:"),
      `restore omitting a past anchor should warn: ${txt(gapRestore)}`,
    );
    const newFill = await B.call("mj_add_comment", {
      comments: [{ anchor: round0Anchor, text: commentText(round0Anchor) }],
    });
    assert(
      newFill.isError && txt(newFill).includes("replace-only"),
      `newly filling a past gap must be replace-only-rejected: ${txt(newFill)}`,
    );
    const clean = await B.call("mj_restore_state", {
      focus: "2",
      comments: restoreComments(r01.map((b) => b.id)),
    });
    assert(
      !clean.isError && txt(clean).includes("restored:") && !txt(clean).includes("warning:"),
      `a full restore should come back clean: ${txt(clean)}`,
    );
  } finally {
    await A.close().catch(() => {}); // double-close after the simulated crash is fine
    if (B) await B.close().catch(() => {});
    await Deno.remove(tmp, { recursive: true }).catch(() => {});
  }
});
