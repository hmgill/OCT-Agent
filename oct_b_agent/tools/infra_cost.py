"""
tools/infra_cost.py
===================
Cost model for an MCP call.

Unlike an LLM call (priced from tokens), an MCP inference call here costs
*compute time* on two platforms it routes through:

    agent → local wrapper → MCP server on **Prefect Horizon** (CPU: decode /
            validate / resize / dispatch) → **Modal** serverless **GPU** endpoint
            (the actual MIRAGE / LO-VLM inference) → back

So per call:

    modal_usd   = gpu_seconds      × modal_rate[gpu_type] × modal_multiplier
    horizon_usd = non_gpu_seconds  × horizon_rate
    total_usd   = modal_usd + horizon_usd

``gpu_seconds`` comes from the Modal worker's reported ``elapsed_s`` when the
tool returns it (LO-VLM does). When it doesn't (MIRAGE currently omits it), we
upper-bound by attributing the whole measured wall time to the GPU leg and flag
the call ``estimated`` — add ``elapsed_s`` to the MIRAGE worker response to make
the split exact.

Rates (captured 2026-06 — verify and override in config/env):
  * Modal A10G  ≈ $0.000306 / GPU-second   (MIRAGE worker)
  * Modal T4    ≈ $0.000164 / GPU-second   (LO-VLM worker)
  * Modal multipliers: region 1.25×–2.5×, non-preemptible 3× (default 1.0 here)
  * Horizon: "pay only for compute you use"; no public per-second rate, so the
    Horizon CPU rate is a configurable placeholder (default ≈ Modal's CPU base
    $0.0000131 / core-second). Set it from your Horizon plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Modal GPU $/second by GPU type (base, pre-multiplier).
MODAL_GPU_RATES: dict[str, float] = {
    "a10g": 0.000306,
    "t4":   0.000164,
    "l4":   0.000222,
    "a100": 0.001036,
    "h100": 0.002778,
}

DEFAULT_HORIZON_RATE_PER_S = 0.0000131   # placeholder CPU core-second; override


@dataclass
class ServerCostSpec:
    key: str
    gpu_type: str = "a10g"
    modal_rate_per_s: Optional[float] = None     # overrides MODAL_GPU_RATES[gpu_type]
    modal_multiplier: float = 1.0                # region × preemption, etc.

    def gpu_rate(self) -> float:
        if self.modal_rate_per_s is not None:
            return self.modal_rate_per_s
        return MODAL_GPU_RATES.get(self.gpu_type.lower(), MODAL_GPU_RATES["a10g"])


@dataclass
class CallCost:
    server: str
    tool: str
    wall_s: float
    gpu_s: float
    horizon_s: float
    modal_usd: float
    horizon_usd: float
    total_usd: float
    gpu_type: str
    estimated: bool          # True when gpu_s was inferred from wall time

    def as_dict(self) -> dict:
        return {
            "server": self.server, "tool": self.tool,
            "wall_s": round(self.wall_s, 4), "gpu_s": round(self.gpu_s, 4),
            "horizon_s": round(self.horizon_s, 4), "gpu_type": self.gpu_type,
            "modal_usd": round(self.modal_usd, 6),
            "horizon_usd": round(self.horizon_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "estimated": self.estimated,
        }


@dataclass
class InfraCostModel:
    servers: dict[str, ServerCostSpec] = field(default_factory=dict)
    horizon_rate_per_s: float = DEFAULT_HORIZON_RATE_PER_S

    def price(self, server_key: str, tool: str, wall_s: float,
              gpu_s: Optional[float]) -> CallCost:
        spec = self.servers.get(server_key) or ServerCostSpec(key=server_key)
        estimated = gpu_s is None
        if estimated:
            gpu_s = max(0.0, wall_s)               # upper-bound: all time on GPU
            horizon_s = 0.0
        else:
            gpu_s = max(0.0, float(gpu_s))
            horizon_s = max(0.0, wall_s - gpu_s)   # remainder = Horizon CPU leg

        modal_usd = gpu_s * spec.gpu_rate() * spec.modal_multiplier
        horizon_usd = horizon_s * self.horizon_rate_per_s
        return CallCost(
            server=server_key, tool=tool, wall_s=wall_s, gpu_s=gpu_s,
            horizon_s=horizon_s, modal_usd=modal_usd, horizon_usd=horizon_usd,
            total_usd=modal_usd + horizon_usd, gpu_type=spec.gpu_type,
            estimated=estimated,
        )


def summarize_infra(costs: list[CallCost]) -> dict:
    """Roll up a list of per-call costs for the run manifest."""
    total = sum(c.total_usd for c in costs)
    modal = sum(c.modal_usd for c in costs)
    horizon = sum(c.horizon_usd for c in costs)
    by_tool: dict[str, dict] = {}
    for c in costs:
        row = by_tool.setdefault(c.tool, {"calls": 0, "usd": 0.0, "gpu_s": 0.0})
        row["calls"] += 1
        row["usd"] = round(row["usd"] + c.total_usd, 6)
        row["gpu_s"] = round(row["gpu_s"] + c.gpu_s, 4)
    return {
        "calls": len(costs),
        "total_usd": round(total, 6),
        "modal_usd": round(modal, 6),
        "horizon_usd": round(horizon, 6),
        "by_tool": by_tool,
        "per_call": [c.as_dict() for c in costs],
    }
