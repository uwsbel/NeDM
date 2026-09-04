"""Chrono Go2 tracking env: the sim-to-sim transfer test for a policy trained in the NRD model.

A SUBCLASS OF HMMWVChronoTrackingEnv, for the same reason go2_tracking_env.py
subclasses the NN env. References, warm start, observations, reward, termination,
episode bookkeeping and the rsl_rl interface are robot-agnostic in that base; what
differs is how one sim is BUILT, STEPPED and READ. Those are five methods.

Pairing matters here more than anywhere else in the pipeline: the whole claim
under test is that a policy trained inside the frozen model transfers to Chrono,
and that claim is only meaningful if the two envs agree on everything EXCEPT the
dynamics. Sharing the observation and reward code by inheritance is what makes
them agree by construction rather than by inspection.

THREE THINGS THE HMMWV BASE DOES NOT KNOW ABOUT A QUADRUPED:

1. THE ACTION IS A VELOCITY COMMAND FEEDING A LOW-LEVEL POLICY, NOT A DRIVER
   INPUT. The RL action is (cmd_vx, cmd_vy, cmd_wz). It is consumed by the
   imported command-conditioned locomotion policy at control_hz, which emits
   twelve joint targets, which a PD servo turns into torques every physics step.
   Three nested rates, and getting any of them wrong changes the plant rather
   than raising.

2. "chrono_step_size_s" MUST BE THE EXCHANGE STEP, NOT simulation.step_size_s.
   The collector integrates at step_size_s = 5e-4 but calls DoStepDynamics with
   the EXCHANGE step of 2e-3 (nedm.quadruped collectors: exchange = exchange_mult
   * step_size_s). The base computes chrono_steps_per_nn_step = dt_s /
   chrono_step_size_s and then calls _advance_sim_steps with that count, so
   passing 5e-4 would ask for 20 advances of 2e-3 each -- 40 ms of simulated time
   per 10 ms NN step, a 4x time dilation that no assertion anywhere would catch.
   _load_chrono_config substitutes the exchange step deliberately.

3. THE ROBOT MUST STAND BEFORE IT CAN WALK. The HMMWV is spawned at a pose with a
   forward velocity and is immediately drivable. The Go2 is spawned above the
   ground in its URDF pose and needs the collector's pose ramp and settle hold
   (0.75 s + 0.5 s) before any locomotion policy output means anything. That runs
   inside _create_sim, BEFORE the base's reference pre-roll, so the pre-roll sees
   a standing robot exactly as the collection did.

CRM eval is minutes per reference: every substep is an SPH solve. Rigid first.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from nedm.rl.go2_tracking_env import go2_default_env_cfg
from nedm.rl.hmmwv_chrono_tracking_env import HMMWVChronoTrackingEnv, default_chrono_env_cfg


@dataclass
class ChronoGo2Sim:
    """One Chrono Go2 world. Mirrors ChronoHMMWVSim's role, not its fields.

    ``command`` stands in for the HMMWV's DriverInputs: it is the mutable slot
    _set_driver_action_np writes and the low-level policy reads.
    """

    system: Any
    robot: Any
    policy: Any
    terrain: Any            # None on rigid ground; the CRMTerrain otherwise
    soil_top_m: float
    command: np.ndarray
    coupled: Any = None
    # Substeps executed since construction. The low-level policy fires on a fixed
    # cadence, and that cadence has to be CONTINUOUS across _advance_sim_steps
    # calls -- resetting it per call would re-trigger the policy at the start of
    # every NN step and change the control rate.
    substep: int = 0
    soil_z: Any = None
    soil_ctrl: float = float("nan")
    sph_probe: Any = None
    # Only for the reward/state capture, which wants the last joint targets.
    joint_targets: Any = field(default=None)


def go2_default_chrono_env_cfg() -> dict[str, Any]:
    """The NN Go2 env's config, plus the Chrono-side keys.

    Starts from go2_default_env_cfg() so the reward, action bounds, episode
    length and termination are THE SAME OBJECTS as the training env's. If they
    are ever allowed to drift, a transfer gap stops being attributable to the
    dynamics.
    """
    cfg = default_chrono_env_cfg()
    cfg.update(go2_default_env_cfg())
    cfg.update(
        {
            "num_envs": 1,
            # Chrono is CPU-bound and serial here; one world at a time.
            "device": "cpu",
            "chrono_config": None,       # required: the collector config to reproduce
            "imported_policy_ckpt": None,  # required: go2_cts_150k.pt
            "urdf_assets_root": "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL",
            "pose_ramp_seconds": 0.75,
            "settle_seconds": 0.5,
            # Matches the collection: foot_margin_spacings * spacing + leg reach.
            "spawn_clearance_m": None,
            "foot_margin_spacings": 1.4,
            "solver_iterations": 150,
            "warm_start_context": True,
        }
    )
    return cfg


class Go2ChronoTrackingEnv(HMMWVChronoTrackingEnv):
    """Chrono Go2 velocity-command tracking on RIGID ground."""

    # ---- configuration -----------------------------------------------------

    def _load_chrono_config(self, config_path: Path) -> dict[str, Any]:
        import json

        import pychrono as chrono

        config = json.loads(Path(config_path).read_text())
        simulation = config.setdefault("simulation", {})
        # SEE POINT 2 IN THE MODULE DOCSTRING. The base divides dt_s by
        # simulation.step_size_s to decide how many times to call
        # _advance_sim_steps, and _advance_sim_steps advances by the EXCHANGE
        # step. These must be the same quantity or simulated time dilates
        # silently.
        exchange = simulation.get("exchange_step_s")
        if exchange is None:
            raise KeyError(
                f"{config_path} has no simulation.exchange_step_s. The Go2 collector "
                "integrates at step_size_s but exchanges at exchange_mult * step_size_s, "
                "and it is the exchange step that DoStepDynamics advances."
            )
        simulation["raw_step_size_s"] = simulation.get("step_size_s")
        simulation["step_size_s"] = float(exchange)
        # configure_chrono_data_paths (called by the base) needs these two keys;
        # the quadruped collector config carries neither because it chdir's into
        # the URDF directory instead. Point them at the installed pychrono tree.
        installed = Path(chrono.GetChronoDataPath())
        config.setdefault("chrono_data_root", str(installed))
        config.setdefault("vehicle_data_root", str(installed / "vehicle"))
        return config

    # ---- world construction ------------------------------------------------

    def _create_sim(self, reference_id: int, ref_step: int) -> ChronoGo2Sim:
        import pychrono as chrono

        from nedm.quadruped import soilprobe
        from nedm.quadruped.constants import GRAVITY, STAND_ACTION
        from nedm.quadruped.robot import Go2Robot

        cfg = self.cfg
        ref_pose = self.reference_set.poses[reference_id, ref_step]

        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(int(cfg["solver_iterations"]))
        # The collector sets these globally before building the robot; collision
        # envelope and margin change contact behaviour, so they are part of the
        # plant, not a rendering detail.
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

        urdf = self._urdf_path()
        soil_top = self._soil_top_m()
        spawn_z = self._spawn_z(chrono, urdf, soil_top)

        init = chrono.ChFramed(
            chrono.ChVector3d(float(ref_pose[0]), float(ref_pose[1]), spawn_z),
            chrono.QuatFromAngleZ(float(ref_pose[2])),
        )
        # Go2Robot resolves URDF-relative mesh paths against the CWD, exactly as
        # the collector does. Restored in a finally so a construction failure
        # does not leave the process in the asset tree.
        cwd = os.getcwd()
        os.chdir(urdf.parent)
        try:
            robot = Go2Robot(system, urdf, init, actuation="torque")
        finally:
            os.chdir(cwd)

        terrain, coupled = self._create_terrain(system, robot, reference_id)
        policy = self._create_policy()
        sim = ChronoGo2Sim(
            system=system,
            robot=robot,
            policy=policy,
            terrain=terrain,
            soil_top_m=soil_top,
            command=np.zeros(3, dtype=np.float32),
            coupled=coupled,
            sph_probe=soilprobe.bind_probe(terrain),
        )
        self._stand_up(sim, STAND_ACTION)
        return sim

    def _create_terrain(self, system: Any, robot: Any, reference_id: int) -> tuple[Any, Any]:
        """Rigid ground. Returns (terrain, coupled); terrain is None so that
        _advance_sim_steps steps the SYSTEM, matching the collector's rigid path."""
        import pychrono as chrono

        from nedm.quadruped.terrain import build_rigid_ground

        build_rigid_ground(chrono, system)
        return None, []

    def _create_policy(self) -> Any:
        from nedm.quadruped.imported_policy import ImportedGo2Policy

        ckpt = self.cfg.get("imported_policy_ckpt")
        if not ckpt:
            raise ValueError(
                "cfg['imported_policy_ckpt'] is required: the RL action is a velocity "
                "COMMAND and has no meaning without the low-level policy that consumes it."
            )
        # family=None keeps the excitation schedule OFF. The command is driven by
        # the RL policy through _set_driver_action_np; a family here would fight
        # it, and set_time would overwrite the command every control tick.
        return ImportedGo2Policy(Path(ckpt), command=(0.0, 0.0, 0.0), family=None)

    def _urdf_path(self) -> Path:
        urdf = Path(self.cfg["urdf_assets_root"]) / "data/robot/go2_irrvis/urdf/go2_description.urdf"
        if not urdf.is_file():
            raise FileNotFoundError(f"Go2 URDF not found: {urdf}")
        return urdf

    def _soil_top_m(self) -> float:
        return float(self.chrono_config["terrain"].get("top_z_m", 0.05))

    def _spawn_z(self, chrono: Any, urdf: Path, soil_top: float) -> float:
        clearance = self.cfg.get("spawn_clearance_m")
        if clearance is not None:
            return soil_top + float(clearance)
        # Same derivation as the collector: enough clearance that no foot starts
        # inside the bed. measure_leg_reach chdir's, so it is bracketed too.
        cwd = os.getcwd()
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach_compat(chrono, urdf)
        finally:
            os.chdir(cwd)
        spacing = float(self.chrono_config["terrain"].get("initial_spacing_m", 0.02))
        return soil_top + float(self.cfg["foot_margin_spacings"]) * spacing + leg_reach

    def _stand_up(self, sim: ChronoGo2Sim, stand_action: np.ndarray) -> None:
        """Pose ramp then settle hold, as the collection did before recording.

        The collector DISCARDS this interval from the dataset, so the model has
        never seen it and the reference has no actions for it. Running it here
        rather than skipping it is what makes the base class's reference pre-roll
        start from the same standing state the collection's first recorded row
        was taken from.
        """
        exchange = self.chrono_step_size_s
        ramp_s = float(self.cfg["pose_ramp_seconds"])
        settle_s = float(self.cfg["settle_seconds"])
        control_every = self._control_every()

        q0 = sim.robot.joint_pos().astype(np.float64)
        sim.robot.actuate(q0)
        num_steps = int(round((ramp_s + settle_s) / exchange))
        for i in range(num_steps):
            t = i * exchange
            if i % control_every == 0:
                if t < ramp_s:
                    alpha = t / max(ramp_s, 1e-9)
                    target = q0 + alpha * (stand_action - q0)
                else:
                    target = np.asarray(stand_action, dtype=np.float64)
                sim.robot.actuate(target)
                sim.joint_targets = target
            self._physics_substep(sim)
        sim.substep = 0

    # ---- stepping ----------------------------------------------------------

    def _control_every(self) -> int:
        control_hz = float(self.chrono_config["simulation"]["control_hz"])
        return max(1, int(round((1.0 / control_hz) / self.chrono_step_size_s)))

    def _physics_substep(self, sim: ChronoGo2Sim) -> None:
        """One exchange step. Rigid steps the system; CRM overrides this."""
        sim.robot.apply_pd()      # every physics step, not every control step
        sim.system.DoStepDynamics(self.chrono_step_size_s)

    def step_raw_actions(self, commands: torch.Tensor):
        """step() with the policy action scaling bypassed.

        FOR THE REPLAY BASELINE ONLY. step() runs policy outputs through
        _scale_policy_actions, which is tanh-squash, recentre on the dataset
        mean, rescale and clamp. A REFERENCE action is already a command in
        m/s and rad/s; pushing it through that transform would evaluate the
        baseline on a different command series than the one the collection
        recorded, and the baseline's entire job is to be that series.

        NOT CLAMPED to action_low/high on purpose. If a recorded command lies
        outside the policy's action range that is a fact worth seeing --
        it means the policy could not have issued it -- and silently clamping
        would hide it inside a number labelled "what the recorded commands did".
        """
        self._raw_action_passthrough = True
        try:
            return self.step(commands)
        finally:
            self._raw_action_passthrough = False

    def _scale_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_raw_action_passthrough", False):
            return policy_actions
        return super()._scale_policy_actions(policy_actions)

    def _set_driver_action_np(self, sim: ChronoGo2Sim, action: np.ndarray) -> None:
        # The command is HELD until the next policy step, exactly as the
        # collection held one family value across its record cadence.
        sim.command = np.asarray(action, dtype=np.float32).copy()

    def _advance_sim_steps(self, sim: ChronoGo2Sim, num_chrono_steps: int) -> None:
        from nedm.quadruped import soilprobe

        if self.vis is not None and sim is self._render_sim:
            raise NotImplementedError("Go2 Chrono eval rendering is not supported; see start_render().")
        control_every = self._control_every()
        for _ in range(num_chrono_steps):
            if sim.substep % control_every == 0:
                # The RL action reaches the plant HERE and nowhere else.
                sim.policy.command = np.asarray(sim.command, dtype=np.float32)
                targets = sim.policy.act(sim.robot)
                sim.robot.actuate(targets)
                sim.joint_targets = targets
                if sim.sph_probe is not None:
                    sim.soil_z, sim.soil_ctrl = soilprobe.sample(sim.sph_probe, sim.robot)
            self._physics_substep(sim)
            sim.substep += 1

    # ---- reading -----------------------------------------------------------

    def _capture_state_pose_np(self, sim: ChronoGo2Sim | None) -> tuple[np.ndarray, np.ndarray]:
        import pychrono as chrono

        from nedm.quadruped.constants import FOOT_BODIES
        from nedm.quadruped.dataset import capture_row

        if sim is None:
            raise RuntimeError("Chrono simulation is not initialized")
        time_s = float(sim.system.GetChTime())
        soil_z = sim.soil_z if sim.soil_z is not None else [float("nan")] * len(FOOT_BODIES)
        row = capture_row(
            chrono=chrono,
            robot=sim.robot,
            terrain=sim.terrain,
            soil_top_m=sim.soil_top_m,
            action=sim.joint_targets,
            command=tuple(float(v) for v in sim.command),
            soil_z=soil_z,
            soil_ctrl=sim.soil_ctrl,
            scenario_name="rl_tracking",
            scenario_family="rl_tracking",
            episode_id="rl_tracking",
            split="eval",
            sample_index=0,
            time_s=time_s,
        )
        state = np.asarray([float(row[field_name]) for field_name in self.state_fields], dtype=np.float32)
        pose = np.asarray(
            [float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["yaw_rad"])],
            dtype=np.float32,
        )
        return state, pose

    # ---- rendering ---------------------------------------------------------

    def start_render(self, reference_id: int | None = None, output_dir: str | Path | None = None) -> Path:
        # The HMMWV path renders with Irrlicht against the vehicle's visual
        # assets and draws the reference line on the RigidTerrain ground body.
        # The Go2 collector renders through a separate VSG plugin
        # (nedm.quadruped.camera); porting it to the offscreen eval path is
        # future work rather than a small change.
        raise NotImplementedError(
            "Go2 Chrono tracking eval does not support rendering yet. "
            "See scripts/collection/collect_go2_smoke.py for the collector's render path."
        )

    def _add_reference_line(self, sim: ChronoGo2Sim, reference_id: int) -> None:  # pragma: no cover
        raise NotImplementedError("Go2 Chrono tracking eval does not support rendering yet.")


class Go2ChronoCRMTrackingEnv(Go2ChronoTrackingEnv):
    """Chrono Go2 tracking on CRM (deformable SPH soil).

    Only the terrain and the substep differ. As on the HMMWV CRM path, the
    terrain co-advances the registered multibody system, so stepping the system
    as well would integrate the robot twice.

    THE SPAWN MUST BE ON THE BED, and the bed is NOT centred on the origin:
    build_crm places the patch at x = patch_x/2 - 0.6, spanning
    [-0.6, patch_x - 0.6]. Reference segments are random mid-episode windows, so
    their start poses are wherever the collected episode happened to be. A spawn
    off the near edge produces an episode with zero foot load that looks like a
    tracking failure -- 21 of those went unnoticed once already in collection.
    Asserted at construction instead.
    """

    BED_MARGIN_M = 0.5

    def _create_terrain(self, system: Any, robot: Any, reference_id: int) -> tuple[Any, Any]:
        import pychrono as chrono
        import pychrono.fsi as fsi
        import pychrono.vehicle as veh

        from nedm.quadruped.terrain import build_crm

        args = self._crm_args()
        # Widen the patch to the bed this episode was actually collected on,
        # BEFORE the assertion, so the assertion tests the bed that will be built.
        bed = self._episode_bed(reference_id)
        if bed is not None:
            x_lo, x_hi, y_lo, y_hi = bed
            args.patch_x = float(x_hi - x_lo)
            args.patch_y = float(y_hi - y_lo)
        self._assert_spawn_on_bed(reference_id, args)
        return build_crm(chrono, fsi, veh, system, robot, args)

    def _episode_bed(self, reference_id: int) -> tuple[float, float, float, float] | None:
        """The bed THIS reference's episode was collected on, from its metadata.

        THE COLLECTION DID NOT USE ONE BED SIZE. 133 CRM episodes ran on
        patch_y 4.0 and 19 on patch_y 8.0 -- collect_go2_smoke's spawn_for uses a
        wider patch for the lateral family, which strafes. The merged
        collector_config.resolved.json records ONE episode's config, so building
        every eval bed from it puts 19 of 152 episodes on a bed half the width
        they were collected on, and a lateral reference starting at |y| = 3.0
        then spawns a metre off the edge and records zero foot load. That is the
        failure the collection already had once with 21 episodes.

        Falls back to the config when an episode's metadata cannot be found, and
        SAYS SO, because silently using the config is how the wrong bed gets
        built without anyone noticing.
        """
        import json

        episode_id = self.reference_set.episode_ids[reference_id]
        for root in self._episode_metadata_roots():
            path = Path(root) / "episodes" / f"{episode_id}.json"
            if path.is_file():
                bed = json.loads(path.read_text()).get("plant_bed_m")
                if bed and len(bed) == 4:
                    return tuple(float(v) for v in bed)
        print(f"[go2-crm] WARNING: no plant_bed_m for {episode_id}; "
              f"falling back to the config bed, which may be the wrong width")
        return None

    def _episode_metadata_roots(self) -> list[str]:
        roots = self.cfg.get("episode_metadata_roots")
        if roots:
            return [str(r) for r in roots]
        return [str(Path.home() / "sbel-artifacts/datasets/go2_merged/crm"),
                str(Path.home() / "sbel-artifacts/datasets/go2_merged/flat")]

    def _crm_args(self) -> SimpleNamespace:
        """Rebuild the namespace build_crm reads, from the collector config.

        Read from the config the DATASET was generated with rather than from
        defaults, so the eval soil is the soil the model was trained on. A
        mismatch here would show up as a transfer gap and be attributed to the
        model.
        """
        terrain_cfg = self.chrono_config["terrain"]
        simulation = self.chrono_config["simulation"]
        soil = terrain_cfg.get("soil") or {}
        return SimpleNamespace(
            spacing=float(terrain_cfg["initial_spacing_m"]),
            # The CFD step is the RAW integration step, not the exchange step --
            # _load_chrono_config overwrote step_size_s, so read what it saved.
            step=float(simulation.get("raw_step_size_s") or simulation["step_size_s"]),
            soil=str(terrain_cfg.get("soil_preset", "training")),
            soil_young=float(soil["young"]) if "young" in soil else None,
            soil_cohesion=float(soil["cohesion"]) if "cohesion" in soil else None,
            artificial_viscosity=float((terrain_cfg.get("sph") or {}).get("artificial_viscosity", 2.0)),
            patch_x=float(terrain_cfg["patch_x_m"]),
            patch_y=float(terrain_cfg["patch_y_m"]),
            depth=float(terrain_cfg["depth_m"]),
            soil_bottom=float(terrain_cfg["bottom_z_m"]),
            no_calf_fsi=False,
            check_embedded=False,
        )

    def _soil_top_m(self) -> float:
        terrain_cfg = self.chrono_config["terrain"]
        if "top_z_m" in terrain_cfg:
            return float(terrain_cfg["top_z_m"])
        return float(terrain_cfg["bottom_z_m"]) + float(terrain_cfg["depth_m"])

    def _assert_spawn_on_bed(self, reference_id: int, args: SimpleNamespace) -> None:
        pose = self.reference_set.poses[reference_id, 0]
        centre_x = args.patch_x / 2 - 0.6
        x_lo, x_hi = centre_x - args.patch_x / 2, centre_x + args.patch_x / 2
        y_lo, y_hi = -args.patch_y / 2, args.patch_y / 2
        margin = self.BED_MARGIN_M
        x, y = float(pose[0]), float(pose[1])
        if not (x_lo + margin <= x <= x_hi - margin and y_lo + margin <= y <= y_hi - margin):
            raise ValueError(
                f"reference {reference_id} ({self.reference_set.episode_ids[reference_id]}) "
                f"starts at ({x:.3f}, {y:.3f}), outside the CRM bed "
                f"[{x_lo:.2f}, {x_hi:.2f}] x [{y_lo:.2f}, {y_hi:.2f}] with {margin} m margin. "
                "A spawn off the bed records zero foot load and reads as a tracking failure."
            )

    def _physics_substep(self, sim: ChronoGo2Sim) -> None:
        sim.robot.apply_pd()
        # CRMTerrain.DoStepDynamics co-steps the coupled FSI + multibody system.
        # Do NOT also step sim.system -- that integrates the robot twice.
        sim.terrain.DoStepDynamics(self.chrono_step_size_s)


def measure_leg_reach_compat(chrono: Any, urdf: Path) -> float:
    """Thin indirection so _spawn_z stays readable and the import stays lazy."""
    from nedm.quadruped.terrain import measure_leg_reach

    return float(measure_leg_reach(chrono, urdf))
