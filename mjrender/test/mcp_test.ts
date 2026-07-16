// End-to-end MCP test: spawn src/mcp.ts over stdio and drive a real JSON-RPC
// exchange (initialize → tools/list → tools/call get_snapshot).

const ROOT = new URL("../", import.meta.url).pathname;
const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

// deno-lint-ignore no-explicit-any
type Json = any;

class McpClient {
  #proc: Deno.ChildProcess;
  #writer: WritableStreamDefaultWriter<Uint8Array>;
  #lines: AsyncIterator<string>;

  constructor() {
    this.#proc = new Deno.Command("deno", {
      args: ["run", "--allow-read", "--allow-write", "--allow-env=HOME", "src/mcp.ts"],
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

  async request(id: number, method: string, params: Json): Promise<Json> {
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

Deno.test("mcp: stateful open_log → snapshot → add_comment → weave end-to-end", async () => {
  const c = new McpClient();
  try {
    const init = await c.request(1, "initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "mjrender-test", version: "0.0.0" },
    });
    if (init.serverInfo?.name !== "mjrender") {
      throw new Error(`unexpected serverInfo: ${JSON.stringify(init.serverInfo)}`);
    }
    await c.send({ jsonrpc: "2.0", method: "notifications/initialized" });

    const tools = await c.request(2, "tools/list", {});
    const names = tools.tools.map((t: Json) => t.name).sort();
    const want = [
      "mj_add_comment",
      "mj_add_note",
      "mj_draft_status",
      "mj_get_final_standings",
      "mj_get_kyoku_result",
      "mj_get_kyoku_start",
      "mj_get_riichi_declarations",
      "mj_get_snapshot",
      "mj_list_anchors",
      "mj_open_log",
      "mj_render_game",
      "mj_render_kyoku",
      "mj_weave_commentary",
    ];
    if (JSON.stringify(names) !== JSON.stringify(want)) {
      throw new Error(`tool set mismatch: ${names}`);
    }

    // only mj_open_log takes a path; everything else errors before a log is open
    const early = await c.request(3, "tools/call", {
      name: "mj_list_anchors",
      arguments: {},
    });
    if (!early.isError || !early.content[0].text.includes("mj_open_log")) {
      throw new Error(`expected 'no log loaded' error, got: ${early.content[0].text}`);
    }

    const opened = await c.request(4, "tools/call", {
      name: "mj_open_log",
      arguments: { path: SAMPLE },
    });
    if (opened.isError || !opened.content[0].text.includes("anchors:")) {
      throw new Error(`unexpected open_log reply: ${opened.content[0].text}`);
    }

    const anchors = await c.request(5, "tools/call", {
      name: "mj_list_anchors",
      arguments: {},
    });
    const table: string = anchors.content[0].text;
    const riichiId = Number(
      table.split("\n").find((l) => l.includes("リーチ判断"))?.match(/^#(\d+)/)?.[1],
    );
    if (!riichiId) throw new Error("no riichi anchor found via MCP");

    // mj_render_game returns the crude outline, not the full transcript
    const outline = await c.request(19, "tools/call", {
      name: "mj_render_game",
      arguments: {},
    });
    const ol: string = outline.content[0].text;
    if (outline.isError || !ol.includes("【東1局") || !ol.includes("◆終局")) {
      throw new Error(`unexpected outline: ${ol.slice(0, 200)}`);
    }
    if (!ol.includes(`〔解説ポイント#${riichiId}:`)) {
      throw new Error("outline missing the riichi anchor in its index");
    }
    const olBody = ol.slice(ol.indexOf("【")); // skip the legend, which mentions the markers
    if (olBody.includes("◆配牌") || olBody.includes("┗")) {
      throw new Error("outline should not contain per-turn detail");
    }

    const snap = await c.request(6, "tools/call", {
      name: "mj_get_snapshot",
      arguments: { anchor: riichiId },
    });
    const text: string = snap.content[0].text;
    for (const needle of ["┌盤面", "リーチ(", "残り山", "手牌 P0:"]) {
      if (!text.includes(needle)) throw new Error(`snapshot missing ${needle}:\n${text}`);
    }

    const bad = await c.request(7, "tools/call", {
      name: "mj_get_snapshot",
      arguments: { anchor: 99999 },
    });
    if (!bad.isError) throw new Error("expected isError for unknown anchor");

    // a structured fact tool round-trips as parseable JSON
    const facts = await c.request(8, "tools/call", {
      name: "mj_get_riichi_declarations",
      arguments: {},
    });
    const decls = JSON.parse(facts.content[0].text);
    if (!Array.isArray(decls) || decls.length === 0 || typeof decls[0].waits !== "string") {
      throw new Error(`unexpected riichi facts: ${facts.content[0].text}`);
    }
    if (decls[0].anchor !== riichiId) {
      throw new Error(`first riichi anchor ${decls[0].anchor} != list_anchors' ${riichiId}`);
    }

    // weaving an empty draft is an error, not an empty document
    const tmp = await Deno.makeTempDir();
    try {
      const out = `${tmp}/woven.txt`;
      const premature = await c.request(9, "tools/call", {
        name: "mj_weave_commentary",
        arguments: { out },
      });
      if (!premature.isError || !premature.content[0].text.includes("mj_add_comment")) {
        throw new Error(`expected empty-draft error: ${premature.content[0].text}`);
      }

      // comments accumulate server-side; a call may batch several anchors
      const added = await c.request(10, "tools/call", {
        name: "mj_add_comment",
        arguments: { comments: [{ anchor: riichiId, text: "MCPテスト解説。" }] },
      });
      const progress: string = added.content[0].text;
      if (added.isError || !progress.includes(`saved #${riichiId}`) || !progress.includes("1/")) {
        throw new Error(`unexpected add_comment reply: ${progress}`);
      }
      // a batch with any bad entry is rejected atomically — nothing is saved
      const badAdd = await c.request(11, "tools/call", {
        name: "mj_add_comment",
        arguments: {
          comments: [{ anchor: 1, text: "巻き添え。" }, { anchor: 99999, text: "范囲外。" }],
        },
      });
      if (!badAdd.isError) throw new Error("expected isError for out-of-range anchor");

      const status = await c.request(12, "tools/call", {
        name: "mj_draft_status",
        arguments: {},
      });
      const checklist: string = status.content[0].text;
      if (!checklist.includes(`✓ #${riichiId}`) || !checklist.includes("・ #")) {
        throw new Error(`draft status should mark filled/unfilled anchors:\n${checklist}`);
      }
      if (!checklist.includes("1/")) {
        throw new Error(`rejected batch must not partially save (#1 leaked):\n${checklist}`);
      }

      // weave writes the accumulated draft to a file; only a summary comes back
      const wove = await c.request(13, "tools/call", {
        name: "mj_weave_commentary",
        arguments: { out },
      });
      const summary: string = wove.content[0].text;
      if (!summary.includes(out) || summary.includes("MCPテスト解説。")) {
        throw new Error(`weave should return a summary, not the document: ${summary}`);
      }
      const doc = await Deno.readTextFile(out);
      if (!doc.includes("◆解説（リーチ判断）: MCPテスト解説。")) {
        throw new Error("woven document missing the spliced comment");
      }
      if (doc.includes(`〔解説ポイント#${riichiId}:`)) {
        throw new Error("filled anchor placeholder should be gone");
      }

      // reopening the SAME log keeps the draft; a relative `out` resolves
      // beside the log file, not the server's cwd
      const localLog = `${tmp}/sample.xml`;
      await Deno.copyFile(SAMPLE, localLog);
      await c.request(14, "tools/call", {
        name: "mj_open_log",
        arguments: { path: localLog },
      });
      const fresh = await c.request(15, "tools/call", {
        name: "mj_draft_status",
        arguments: {},
      });
      if (!fresh.content[0].text.includes("0/")) {
        throw new Error(`opening another log should start an empty draft: ${fresh.content[0].text}`);
      }
      await c.request(16, "tools/call", {
        name: "mj_add_comment",
        arguments: { comments: [{ anchor: riichiId, text: "相対パステスト。" }] },
      });
      const reopened = await c.request(17, "tools/call", {
        name: "mj_open_log",
        arguments: { path: localLog },
      });
      if (!reopened.content[0].text.includes("1/")) {
        throw new Error(`reopening the same log should keep the draft: ${reopened.content[0].text}`);
      }
      const rel = await c.request(18, "tools/call", {
        name: "mj_weave_commentary",
        arguments: { out: "commentary.txt" },
      });
      const relSummary: string = rel.content[0].text;
      if (rel.isError || !relSummary.includes(`${tmp}/commentary.txt`)) {
        throw new Error(`relative out should resolve beside the log: ${relSummary}`);
      }
      await Deno.readTextFile(`${tmp}/commentary.txt`); // must exist there
    } finally {
      await Deno.remove(tmp, { recursive: true });
    }
  } finally {
    await c.close();
  }
});
