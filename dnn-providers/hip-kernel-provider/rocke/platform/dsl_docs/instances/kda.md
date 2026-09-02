# Chunkwise KDA

Chunkwise gated delta-rule linear-attention prefill is implemented by the
gfx950-only library kernel
[`kda_chunkwise.py`](../../../library/kernels/gfx950/kda_chunkwise.py).
The host builders are in
[`library/builders/gfx950/kda`](../../../library/builders/gfx950/kda/).

## Specs and compositions

`KdaTileSpec` owns the chunk, workgroup, MFMA, solve, padding, and occupancy
knobs. The operation has three build entry points:

- `build_kda_chunk_prep`: constructs the six state-independent tiles for each
  chunk.
- `build_kda_chunk_scan`: stages those tiles and applies the serial state
  recurrence for each `(batch, head)`.
- `build_kda_chunk_fused`: constructs and consumes the tiles in one workgroup.

The split composition is the default. The fused composition is useful for
cross-checking the shared scan body and for future LDS/resource schedules.
Both support an optional initial state and final-state output.

## Numerical contract

The implementation uses a midpoint decay factorization so per-channel gate
ratios remain finite over a chunk. The tile builder and both complete paths are
checked against independent float64 references. The split and fused paths are
also checked for bitwise equality when their accumulation order is identical.

## Validation

Run the CPU admission/build checks from `rocke/library`:

```bash
python -m pytest tests/test_kda_chunkwise_spec.py
```

Run the gfx950 numeric checks on a matching GPU:

```bash
python -m pytest tests/test_kda_chunkwise_gfx950_numeric.py -m gpu
```

Run the CPU wiring checks for the dispatcher family:

```bash
python -m pytest tests/dispatch/kda/test_gfx950_wiring.py
```

## Dispatch

`dispatch.kda` registers one candidate per emitted kernel:

| Candidate | `algorithm` | Kernel |
| --- | --- | --- |
| `kda_gfx950_chunk_fused` | `chunk_fused` | `build_kda_chunk_fused` |
| `kda_gfx950_chunk_prep` | `chunk_prep` | `build_kda_chunk_prep` |
| `kda_gfx950_chunk_scan` | `chunk_scan` | `build_kda_chunk_scan` |

```python
from dispatch.kda import KdaRequest, dispatch_kda

result = dispatch_kda(KdaRequest(batch=8, num_heads=16, seqlen=2048, arch="gfx950"))
```

The fused kernel is the default. The split halves are opt-in by `algorithm`
or `spec_id` and must be launched in that order: the scan consumes the tiles
the tile builder writes. There is no automatic fused/split routing, because
the crossover has been measured at too few shapes to encode a threshold; see
`dispatch/kda/common.py`. `bind` is not wired, so launching still goes through
the host builders.

The standalone benchmark scenario is `benchmarks.gfx950.kda.benchmark_chunkwise`.
