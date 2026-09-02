# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pieces of the KDA family: the request and the gates every candidate
re-uses.

Arch-neutral by construction even though the only arch today is gfx950: the
arch module imports this one and nothing here imports it, which keeps the
registry assembly in ``__init__`` free of import-order dependence.

SCOPE -- what this dispatcher decides
-------------------------------------
Which of the two chunkwise prefill paths runs: the **fused** kernel (one
workgroup per (batch, head) walks that head's chunks, keeping the six
per-chunk tiles in LDS), or the **split** path (a tile-builder kernel that is
one workgroup per chunk, then a state-scan kernel over the materialized
tiles). Both compute the same recurrence -- bitwise, since they share one
emitted scan body -- so the choice trades HBM traffic against parallelism.
See :mod:`kernels.gfx950.kda_chunkwise`.

``head_v`` is the value width of one head and reaches the kernel unchanged.
gfx950's 160 KiB LDS holds a whole DV=128 state mirror, so unlike gfx942 there
is no host-side partition of a logical head into narrower workgroups. The
scan's ``value_splits`` is a spec knob rather than a request property, and its
own grid helper applies it, so ``workgroups`` here counts (batch, head)
recurrences and nothing else.

DEFERRED -- the fused/split crossover
-------------------------------------
There are measurements (MI355X, chunk 32: split 0.151 ms vs fused 0.204 ms at
B=8 H=8 T=1024; split 0.425 ms vs fused 0.405 ms at B=8 H=16 T=2048), and they
point the way the kernel docstrings predict -- the fused kernel needs ``B*H``
comfortably above the CU count to fill the device, because it runs one
workgroup per (batch, head) where the split tile builder runs one per chunk.
One config per shape at three shapes is evidence, not a threshold, so the
split candidates are **opt-in** and fused is the default rather than a
heuristic that would look measured and is not. Naming
``algorithm="chunk_prep"`` / ``"chunk_scan"`` selects the split halves.

Also deferred: ``bind``. These kernels launch through ``KernelLauncher`` with
torch tensors in ``builders/gfx950/kda``, which is a different seam;
:class:`~rocke.dispatch.core.ProblemBinding` also describes a single launch,
and the split path is two of them with a tile workspace in between. So the
registry does not set ``require_binding``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from kernels.gfx950.kda_chunkwise import (
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage as a Capability. Copying the numbers would
    # drift in the direction that fails silently -- the prefilter rejecting a
    # shape the kernel had since learned to run.
    KDA_CHUNK_SIZES,
    KDA_DTYPES,
)
from rocke.core.arch import ArchTarget
from rocke.dispatch.core import KernelCandidate, OperatorRequest

FAMILY = "kda_chunkwise"
KDA_ABI_VERSION = "rocke-kda-chunkwise/v1"


@dataclass(frozen=True)
class KdaRequest(OperatorRequest):
    """Normalized chunkwise Kimi Delta Attention prefill request.

    ``seqlen`` is the padded per-sequence length; this family has no varlen
    path, so a ragged batch must be padded by the caller.

    ``chunk_size`` defaults to the length gfx950 is tuned for. It is a request
    field rather than a spec knob because it changes the arithmetic -- the
    C x C triangular solve against the number of serial scan steps -- and so
    belongs to the problem the caller is asking for.
    """

    batch: int
    num_heads: int
    seqlen: int
    arch: str
    head_k: int = 128
    head_v: int = 128
    chunk_size: int = 32
    op: str = "kda"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    has_initial_state: bool = False
    store_final_state: bool = True

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        return d

    def dims(self) -> dict[str, int]:
        return {
            "batch": int(self.batch),
            "num_heads": int(self.num_heads),
            "seqlen": int(self.seqlen),
            "head_k": int(self.head_k),
            "head_v": int(self.head_v),
            "chunk_size": int(self.chunk_size),
            "num_chunks": self.num_chunks,
        }

    def features(self) -> frozenset[str]:
        active = set()
        if bool(self.has_initial_state):
            active.add("initial_state")
        if bool(self.store_final_state):
            active.add("final_state")
        return frozenset(active)

    @property
    def num_chunks(self) -> int:
        """Chunks per sequence. Zero when ``seqlen`` does not tile exactly."""
        chunk = int(self.chunk_size)
        if chunk <= 0 or int(self.seqlen) % chunk:
            return 0
        return int(self.seqlen) // chunk

    @property
    def workgroups(self) -> int:
        """Independent (batch, head) scan streams.

        The unit of the recurrence: one workgroup owns one of these and walks
        its chunks in order. An arch that cannot fit a whole head in LDS
        multiplies this in its own module rather than here.
        """
        return int(self.batch) * int(self.num_heads)


KDA_DIM_VOCABULARY = (
    "batch",
    "num_heads",
    "seqlen",
    "head_k",
    "head_v",
    "chunk_size",
    "num_chunks",
)

KDA_FEATURES = frozenset({"initial_state", "final_state"})


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, KdaRequest):
        return [f"expected KdaRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "kda":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("batch", "num_heads", "seqlen", "head_k", "head_v", "chunk_size"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _selector_matches(req: KdaRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


__all__ = [
    "FAMILY",
    "KDA_ABI_VERSION",
    "KDA_CHUNK_SIZES",
    "KDA_DIM_VOCABULARY",
    "KDA_DTYPES",
    "KDA_FEATURES",
    "KdaRequest",
    "_request_errors",
    "_selector_matches",
]
