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
      args: ["run", "--allow-read", "src/mcp.ts"],
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

Deno.test("mcp: initialize, list tools, call get_snapshot end-to-end", async () => {
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
    const want = ["get_snapshot", "list_anchors", "render_game", "render_kyoku"];
    if (JSON.stringify(names) !== JSON.stringify(want)) {
      throw new Error(`tool set mismatch: ${names}`);
    }

    const anchors = await c.request(3, "tools/call", {
      name: "list_anchors",
      arguments: { path: SAMPLE },
    });
    const table: string = anchors.content[0].text;
    const riichiId = Number(table.split("\n").find((l) => l.includes("リーチ判断"))?.match(/^#(\d+)/)?.[1]);
    if (!riichiId) throw new Error("no riichi anchor found via MCP");

    const snap = await c.request(4, "tools/call", {
      name: "get_snapshot",
      arguments: { path: SAMPLE, anchor: riichiId },
    });
    const text: string = snap.content[0].text;
    for (const needle of ["┌盤面", "リーチ(", "残り山", "手牌 P0:"]) {
      if (!text.includes(needle)) throw new Error(`snapshot missing ${needle}:\n${text}`);
    }

    const bad = await c.request(5, "tools/call", {
      name: "get_snapshot",
      arguments: { path: SAMPLE, anchor: 99999 },
    });
    if (!bad.isError) throw new Error("expected isError for unknown anchor");
  } finally {
    await c.close();
  }
});
