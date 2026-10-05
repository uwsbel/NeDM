# Pool-ball NRD study

Two balls on a 9-ft table with four cushions. Ball A is launched with
`(vx, vy)` and hits ball B, which then usually rebounds off a cushion. The
study trains a Transformer-backbone NRD with learned contact switches and
bounce networks on Chrono data. It then differentiates the model's rollout to
choose A's launch so that B is at a target position after t = 2.0 s. Chrono
replays every optimised launch.

- [Physics setup and checks](physics.md): why the setup uses penalty contact,
  plus calibration and timestep convergence.
- [Model: networks, I/O and how they connect](model.md)
- [Design notes written before the pilot](design.md)
- [Results](results.md): data, model accuracy on the sealed cohort, targeting
  in Chrono, baselines, costs.

Work tree: `~/NeDM-pool-ball`, branch `study/pool-ball`. Code:
`src/nedm/pool_ball/`, configs: `configs/pool_ball/`, AMD job scripts:
`scripts/pool_ball/cluster/`. AMD root:
`/work1/dannegrut/harry/experiments/pool_ball_20261002`.

## Key artifacts (local, `artifacts/pool_ball/`)

- `data_v1/`: data quality and diversity report (JSON + figure) and a video of sample Chrono shots.
- `architecture/pool_nrd_architecture.png`: the selected model, with each network's inputs and outputs.
- `selected_model/`: frozen checkpoint (`best.pt`, SHA256 a9d01435...), run configuration, training log, validation and gradient check.
- `certification/`: sealed-cohort certification of the 15 frozen candidates, plus their hashes.
- `targeting_sealed/`: 100 sealed targets, with:
  - `optimization.json` (search results);
  - `chrono_verification.json` (Chrono replays);
  - `iteration_vs_loss_{lm,gd}.png`: model distance and Chrono miss against iteration, 10 targets;
  - `iterations_chrono_{lm,gd}.mp4`: Chrono replays of the launch every 5 iterations.
- `targeting_sealed_small_data/`: the post-certification small-data look, with its own hash manifest.

Load the model with `nedm.pool_ball.model.load_pool(path, device)`. Then:
- `model.rollout(initial_state(scene, velocity), 200)` gives the trajectory to 2.0 s;
- `velocity` is `[..., 2]` = `(vx, vy)` and stays differentiable.
