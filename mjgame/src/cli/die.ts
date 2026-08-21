// The one way this CLI stops: a message on stderr and exit code 2.
//
// Its own module because everything else in `cli/` — and the harness the CLI
// drives — needs it, and a leaf keeps the import graph acyclic (`args.ts` reads
// `--ktune` through `harness.ts`, which in turn refuses a bad file with `die`).

export function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}
