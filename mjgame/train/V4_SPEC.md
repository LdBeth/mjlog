# Feature v4: attention river encoder — FROZEN SPEC

Three implementations (TS inference, MLX trainer, native C) must realize this
byte-for-byte identically at the weight-file level and numerically equivalently
at the forward level (tolerance: |Δlogit| < 1e-4, same argmax on fixtures —
sgemv accumulation order already makes bit-exactness impossible, as in v3).
Nothing here is negotiable per-implementation; a change requires editing THIS
file first.

## Token stream

One token per river entry (discard), ALL FOUR seats including self.
Order: seatRel 0 (self) first, then seatRel 1, 2, 3 (relative to the acting
seat, same convention as the planes); within a seat, chronological river order;
each river truncated to its FIRST 24 entries. Max tokens SEQ_MAX = 96.

Packed encoding (the "seq" d-line field, base64 of Int8Array, 4 bytes/token):

    [type, seatRel, idx, flags]

- `type`: tile type 0–33
- `seatRel`: 0–3
- `idx`: 0-based position in that river, 0–23
- `flags`: bit0 = tsumogiri, bit1 = riichi declaration tile, bit2 = called away
  (the entry was claimed into a meld)

Dense expansion (inside each forward implementation, from the packed form):
x = concat(onehot34(type), onehot4(seatRel), idx/24, tsumogiri, riichiDecl,
calledAway) — **42 dims**, in exactly that order.

## Encoder forward

d_model D = 64, heads H = 4, head dim 16, softmax scale 1/4. L = token count.

1. h_i = relu(W_in x_i + b_in)                    h: [L,64]
2. q_i = Wq h_i + bq;  k_i = Wk h_i + bk;  v_i = Wv h_i + bv
3. per head: A = softmax(Q K^T / 4) over the L tokens (full bidirectional,
   no causal mask); O = A V; concat heads → O: [L,64]
4. m_i = Wo O_i + bo
5. α = softmax(u · m_i) over i;  p = Σ_i α_i m_i   (learned-query pooling)
6. z = Wz p + bz                                   z: [64]

**L = 0** (no discards yet): p = 0, therefore z = bz. All implementations must
special-case this identically.

## Weight file `attn.f32`

Same convention as policy.f32: each matrix [out][in] row-major, immediately
followed by its bias; little-endian float32. Order:

    W_in[64][42] b_in[64]
    Wq[64][64] bq[64]  Wk[64][64] bk[64]  Wv[64][64] bv[64]
    Wo[64][64] bo[64]
    u[64]
    Wz[64][64] bz[64]

Total floats: 64*42+64 + 4*(64*64+64) + 64 + 64*64+64 = **23,616** (94,464
bytes). (An earlier revision of this line said 23,872 — that was an arithmetic
error in the prose; the tensor list above was always authoritative. Loaders
must REJECT any other file size rather than misalign tensors.)

Head split: heads are CONTIGUOUS 16-column slices of the 64 (the reshape
(L, 4, 16) convention — matches MLX's natural reshape and the C lda=64 slices).

Out-of-range inputs (all implementations): ntok<0 or NULL tokens ⇒ treat as 0;
ntok>96 ⇒ clamp to 96; a type/seatRel/idx outside its range sets no one-hot
bit. idx/24 is computed in double precision and rounded once to float32.

## Network input layout

- Policy fc1 input: **[planes+scalars 1674][z 64] = 1738**. Planes/scalars keep
  their v3 offsets 0..1673 unchanged; z occupies 1674..1737.
- Critic fc1 input: **[public 1674][oracle 170][z 64] = 1908**. Existing v3
  critic offsets unchanged; z appended at the END.
- fc2/fc3 and the aux rows are untouched (widths 512/256, 79+24 rows).

## Contract changes

- `FEATURES.version` → 4. Planes (48) and scalars (42) are UNCHANGED — v3
  digests must still pass on the prefix. "v":4 on a d-line means the "seq"
  field is present; ppo.py REQUIRES v4 lines (reject v3 datasets with a clear
  error, as v3 did to v2).
- manifest.json gains an `attn` file entry (`"attn": "attn.f32"`) and reports
  version v4. The PRESENCE of `attn` is the sole v4 discriminator — there is
  deliberately no separate `feature_version` field (two discriminators that
  can disagree are worse than one); a 1738-wide fc1 without `attn`, or `attn`
  with a 1674-wide fc1, is a load error. `features` stays
  `{"planes":48,"scalars":42}` byte-for-byte. Loading a v3 manifest must still
  work everywhere (no seq path, 1674-wide fc1) — play and bench against old
  snapshots stay possible.
- An empty river state encodes `"seq"` as the empty string; loaders read a
  MISSING `seq` on a v4 line as L=0 but count and report it (JSON cannot
  distinguish absent from empty, and L=0 is the commonest state in the game).

## Migration (train/widen4.py, v3 → v4)

Function-preserving surgery, the widen.py pattern:
- policy fc1: new [512][1738], columns 0..1673 copied, columns 1674..1737 ZERO.
- critic fc1: new [512][1908], columns 0..1843 copied, columns 1844..1907 ZERO.
- attn.f32: ALL parameters ~ normal(0, 0.02) from --seed (u and biases too).
  Wz/bz are random like the rest — the ZERO side is the consumer columns.
  (Zeroing both sides would be a dead saddle: grad(attn) flows through the
  zero fc1 columns and grad(fc1 cols) is ∝ z, so exactly one side must be
  nonzero for the path to wake. z ≠ 0 + zero columns ⇒ output unchanged AND
  the column gradients are alive.)
- Verify: v3 and v4 policy forwards agree EXACTLY (float64 reference) on
  random inputs; same for critic; idempotence guard (running twice = error),
  auto-verify on by default. Exit codes as widen.py.

## Native C ABI (native/rlnet.c)

Keep the existing MLP entry points working for v3 manifests. Add:

    rlnet_attn *rlnet_attn_create(const char *attn_f32_path);
    void rlnet_attn_destroy(rlnet_attn *);
    /* tokens: packed int8, 4 bytes/token as above; ntok in [0,96].
       Writes the 64-float z into out_z. */
    void rlnet_attn_encode(const rlnet_attn *, const int8_t *tokens,
                           int ntok, float *out_z);

Policy forward stays the generic MLP over the 1738-wide input; net.ts
concatenates z itself and calls the existing forward. cblas_sgemm/sgemv from
Accelerate; hand-rolled softmax (subtract max, expf, normalize — document that
expf parity with TS Math.exp is within the stated tolerance).
