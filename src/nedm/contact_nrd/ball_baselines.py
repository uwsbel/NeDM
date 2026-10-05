"""Score the earlier bouncing-ball models on the new sealed cohort (same metric code as their study)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from nedm.bouncing_ball.model import load_model
from nedm.bouncing_ball.transformer_contact_eval import data_packet, free_metrics

ROOT = Path("/work1/dannegrut/harry/experiments")
MODELS = {  # earlier frozen bouncing-ball models (no retraining)
    "two_switch_two_bounce_k8 (v2 validation pick)": ROOT / "ball_transformer_v2_20261001T214000Z/frozen/trained_ap_k8_w128_l4/best.pt",
    "one_switch_shared_bounce (v2, user's design)": ROOT / "ball_transformer_v2_20261001T214000Z/frozen/direct_binary_mlp_shared5/best.pt",
    "transformer_with_analytic_contact_timing (v2 historical)": ROOT / "ball_span_v1_20260930/runs/nrd_v2_certified/best.pt",
    "analytic_flight_and_timing_plus_mlp (reference)": ROOT / "ball_precision_20261001/runs/certified_v2/best.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="merged certification data (10 ms, split 2 = new sealed cohort)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    torch.set_num_threads(1)
    data = data_packet(args.data, "cuda")
    ids = torch.nonzero(data["splits"] == 2).flatten()
    out = {"episodes": len(ids), "data_sha256": data["index"]["model_data_sha256"], "models": {}}
    for name, path in MODELS.items():
        model, meta = load_model(path, "cuda")
        model.double()
        native = model.dt
        if abs(model.dt - data["dt"]) > 1e-12:
            model.dt = data["dt"]  # timestep-flexible analytic reference, as in its certification
        m = free_metrics(model, data, ids)
        out["models"][name] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "native_dt_s": native, "evaluation_dt_s": model.dt,
                               "rmse_mm": {k: 1e3 * v for k, v in m["position_rmse_m"].items()},
                               "endpoint_mm": {k: 1e3 * v for k, v in m["endpoint_error_m"].items()},
                               "max_error_mm": {k: 1e3 * v for k, v in m["maximum_position_error_m"].items()},
                               "contact_order_fraction": m["contact_order_fraction"], "finite_fraction": m["finite_fraction"],
                               "per_episode": [{"episode_id": data["index"]["episodes"][e["index"]]["episode_id"], "finite": e["finite"],
                                                "rmse_m": e.get("position_rmse_m"), "end_m": e.get("endpoint_error_m"),
                                                "event_ok": e.get("contact_order", False)} for e in m["per_episode"]]}
        print(json.dumps({name: {"rmse_median_mm": out["models"][name]["rmse_mm"]["median"], "rmse_p95_mm": out["models"][name]["rmse_mm"]["p95"],
                                 "endpoint_p95_mm": out["models"][name]["endpoint_mm"]["p95"], "order": m["contact_order_fraction"]}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
