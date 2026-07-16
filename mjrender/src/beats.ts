// Beat (commentary-anchor) enumeration, shared by the transcript and the
// snapshot/query side.
//
// Anchor IDs are assigned during the annotated render (render.ts) — beat
// placement depends on replay-computed values (danger, shanten), so deriving
// the list from the same pass is what guarantees `#N` in the transcript and
// `list_anchors`/`get_snapshot` can never disagree. Beats do not depend on any
// RenderOption; enumeration always uses the defaults.

import type { Beat, Game } from "./model.ts";
import { renderGameAnnotated } from "./render.ts";

export function enumerateBeats(game: Game): Beat[] {
  return renderGameAnnotated(game, { hands: "key" }).beats;
}
