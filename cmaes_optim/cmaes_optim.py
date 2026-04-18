import argparse
import multiprocessing
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import warp as wp
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    import cma
except ImportError as e:
    raise ImportError(
        "The `cma` package is required for zero-order optimisation. Install it via `pip install cma`."
    ) from e


def set_all_seeds(seed: int = 0):
    """Set python / numpy / torch / warp seeds to make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        wp.set_seed(seed)
    except Exception:
        pass


def normalise(value: float, min_val: float, max_val: float) -> float:
    """Normalise *value* to the [0,1] range given the provided bounds."""
    assert min_val < max_val, "Min must be lower than max."
    return (value - min_val) / (max_val - min_val)


def denormalise(value: float, min_val: float, max_val: float) -> float:
    """Inverse of :func:`normalise`."""
    assert 0.0 <= value <= 1.0, "Value must be in [0,1] when denormalising."
    return value * (max_val - min_val) + min_val


def batch_chamfer_distance_single_direction(
    p1: torch.Tensor, p2: torch.Tensor
) -> torch.Tensor:
    """Computes batched single-direction Chamfer distance from p1 to p2.
    Args:
        p1 (torch.Tensor): (B, N, 3)
        p2 (torch.Tensor): (B, M, 3)
    Returns:
        torch.Tensor: (B,) tensor of losses for each batch element.
    """
    p1_expanded = p1.unsqueeze(2)  # (B, N, 1, 3)
    p2_expanded = p2.unsqueeze(1)  # (B, 1, M, 3)
    dist_matrix = torch.sum((p1_expanded - p2_expanded) ** 2, dim=-1)  # (B, N, M)
    min_dist_p1_p2, _ = torch.min(dist_matrix, dim=2)  # (B, N)
    loss_p1_p2 = torch.mean(min_dist_p1_p2, dim=1)
    return loss_p1_p2


# -----------------------------------------------------------------------------
#  Objective function
# -----------------------------------------------------------------------------


class SimulationObjective:
    """Wrapper that evaluates a set of simulation parameters on a dataset sample."""

    def __init__(
        self,
        cfg,
        dataset,
        device: str,
        param_bounds: dict[str, tuple[float, float]],
        sample_size: int = 4,
        seed: int = 0,
        use_graph: bool = True,
        chamfer_weight: float = 1.0,
        track_weight: float = 1.0,
    ) -> None:
        self.base_cfg = cfg
        self.dataset = dataset
        self.device = device
        self.sample_size = sample_size
        self.use_graph = use_graph
        self.chamfer_weight = chamfer_weight
        self.track_weight = track_weight
        self.param_bounds = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in param_bounds.items()
        }
        print(self.param_bounds)
        exit()
        self._rng = np.random.RandomState(seed)
        set_all_seeds(seed)

    def to_search_vector(self, params: dict) -> List[float]:
        """Convert *params* dictionary → normalised vector for CMA-ES."""
        vec: List[float] = []
        for k, (low, high) in self.param_bounds.items():
            v = params[k]
            vec.append(normalise(v, low, high))
        return vec

    def vector_to_params(self, vec: List[float]) -> dict:
        """Inverse of :meth:`to_search_vector`."""
        out = {}
        for idx, (k, (low, high)) in enumerate(self.param_bounds.items()):
            den = denormalise(vec[idx], low, high)
            if k == "max_springs_per_node":
                low = int(low)
                high = int(high)
                den = int(round(den))
                den = max(low, min(high, den))
                out[k] = den
            else:
                out[k] = float(den)
        return out

    def __call__(self, x: List[float]) -> float:
        params = self.vector_to_params(x)
        indices = self._rng.choice(
            len(self.dataset), size=self.sample_size, replace=False
        )

        mean_loss = self._simulate_batch(indices, params)

        if not np.isfinite(mean_loss):
            return 1e8

        torch.cuda.empty_cache()
        return mean_loss

    def _simulate_batch(self, dataset_indices: np.ndarray, params: dict) -> float:
        """Run a *batch* of simulation rollouts and compute mean MSE-x loss."""

        wp.config.verbose = False
        wp.config.quiet = True
        wp.config.print_launches = False

        wp.init()

        wp_device = (
            wp.get_device(self.device)
            if self.device.startswith("cuda")
            else wp.get_device()
        )

        batch_data = [self.dataset[i] for i in dataset_indices]

        init_data, actions, gt_states = zip(*batch_data)

        init_fields = list(zip(*init_data))
        x0_list = list(init_fields[0])
        v0_list = list(init_fields[1])
        fill_points_list = list(init_fields[7]) if len(init_fields) >= 8 else None
        _, grippers_list = zip(*actions)
        gt_x_list, _ = zip(*gt_states)

        original_num_particles = x0_list[0].shape[0]

        if fill_points_list is not None and int(self.base_cfg.sim.num_fill_points) > 0:
            target_fill_points = int(self.base_cfg.sim.num_fill_points)
            new_x0_list: List[torch.Tensor] = []
            new_v0_list: List[torch.Tensor] = []
            for i in range(len(x0_list)):
                x0_i = x0_list[i]
                v0_i = v0_list[i]
                fill_i = fill_points_list[i]

                zeros_vel = torch.zeros((target_fill_points, 3), dtype=v0_i.dtype)
                new_x0_list.append(torch.cat([x0_i, fill_i], dim=0))
                new_v0_list.append(torch.cat([v0_i, zeros_vel], dim=0))

            x0 = torch.stack(new_x0_list)
            v0 = torch.stack(new_v0_list)
        else:
            x0 = torch.stack(x0_list)
            v0 = torch.stack(v0_list)

        grippers = torch.stack(grippers_list)
        gt_x = torch.stack(gt_x_list)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = x0.to(device)
        v0 = v0.to(device)
        if isinstance(grippers, torch.Tensor):
            grippers = grippers.to(device)

        batch_size = len(dataset_indices)
        points_per_env = x0.shape[1]
        num_steps_total = gt_x.shape[1]
        num_particles_orig = original_num_particles

        gripper_data_to_pass = (
            grippers
            if (
                self.base_cfg.sim.gripper_forcing
                and not self.base_cfg.sim.gripper_points
                and isinstance(grippers, torch.Tensor)
                and (grippers.shape[2] if grippers.ndim >= 3 else 0) > 0
            )
            else None
        )
        from meta_material.sim import create_simulator

        sim = create_simulator(
            backend=getattr(self.base_cfg.sim, "backend", "spring"),
            x=x0.detach().cpu().numpy(),
            v=v0.detach().cpu().numpy(),
            grippers=(
                grippers.detach().cpu().numpy()
                if isinstance(grippers, torch.Tensor)
                else None
            ),
            points_per_env=points_per_env,
            batch_size=batch_size,
            threshold=params["threshold"],
            stiffness=params["stiffness"],
            damping=params["damping"],
            mass_per_point=1.0,
            sim_dt=float(self.base_cfg.sim.dt),
            sim_substeps=int(self.base_cfg.sim.sim_substeps),
            device=wp_device,
            visualize=False,
            max_springs_per_node=int(params["max_springs_per_node"]),
            record_trajectory=False,
            ground_friction=params["ground_friction"],
        )

        if not self.use_graph:
            sim.use_cuda_graph = False

        x_sim, v_sim = sim.get_initial_state()
        device = x_sim.device

        sim_x_trajectory: List[torch.Tensor] = []

        for step in range(num_steps_total):
            x_sim, v_sim = sim(
                step,
                x_sim.detach().clone(),
                v_sim.detach().clone(),
                None,
                gripper_data_to_pass,
            )
            sim_x_trajectory.append(x_sim[:, :num_particles_orig])

        sim.save_renderer() if hasattr(sim, "save_renderer") else None
        if sim.record_trajectory:
            sim.save_trajectory("cmaes_optim/outputs/trajectory.npz")
            print("Saved trajectory to cmaes_optim/outputs/trajectory.npz")

        sim_x_trajectory_tensor = torch.stack(sim_x_trajectory, dim=1)

        total_chamfer_loss = 0.0
        total_track_loss = 0.0
        num_steps = sim_x_trajectory_tensor.shape[1]
        gt_x_device = gt_x.to(device)

        for t in range(num_steps):
            sim_x_t = sim_x_trajectory_tensor[:, t, :, :]  # B, N, 3
            gt_x_t = gt_x_device[:, t, :, :]  # B, N, 3

            chamfer_loss_t = batch_chamfer_distance_single_direction(
                sim_x_t, gt_x_t
            ).mean()
            total_chamfer_loss += chamfer_loss_t

            track_loss_t = torch.nn.functional.mse_loss(sim_x_t, gt_x_t)
            total_track_loss += track_loss_t

        avg_chamfer_loss = total_chamfer_loss / num_steps
        avg_track_loss = total_track_loss / num_steps
        loss = (self.chamfer_weight * avg_chamfer_loss) + (
            self.track_weight * avg_track_loss
        )

        del sim
        return loss.item()


def build_cfg(config_path: str, overrides: List[str] | None = None) -> DictConfig:
    """Load the Hydra config located at an arbitrary directory (absolute or relative)."""
    cfg_path = Path(config_path).expanduser().resolve()
    cfg_dir = str(cfg_path.parent)
    cfg_name = cfg_path.stem

    overrides = overrides or []
    with initialize_config_dir(version_base="1.2", config_dir=cfg_dir):
        try:
            cfg = compose(config_name=cfg_name, overrides=overrides)
        except TypeError:
            cfg = compose(config_name=cfg_name, overrides_list=overrides)
    return cfg


def get_param_bounds(cfg: DictConfig) -> dict[str, tuple[float, float]]:
    """Extract CMA-ES parameter bounds from Hydra config."""
    return {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in cfg.cmaes.param_bounds.items()
    }


def resolve_cfg_overrides(name: str, overrides: List[str]) -> List[str]:
    """Auto-select CMA-ES bounds from dataset naming unless explicitly overridden."""
    resolved = list(overrides)
    if any(override.startswith("cmaes_bounds=") for override in resolved):
        return resolved
    if "paper" in name.lower():
        resolved.append("cmaes_bounds=paper")
    return resolved


def get_dataset(
    cfg, dataset_root: Path, source_dataset_root: Path, device: torch.device
):
    """Utility to build a :class:`RealTeleopBatchDataset` for evaluation."""
    from meta_material.data import RealTeleopBatchDataset

    dataset = RealTeleopBatchDataset(
        cfg,
        dataset_root=dataset_root,
        source_data_root=source_dataset_root,
        device=device,
        num_steps=cfg.sim.num_steps_train,
        train=True,
        dataset_non_overwrite=False,
        load_partial_dataset=False,
        lazy_load=True,
    )
    return dataset


def _log_params_callback(
    solver: "cma.CMAEvolutionStrategy", objective: SimulationObjective
) -> None:  # noqa: N801
    print(f"\n--- CMA-ES Iteration {solver.countiter} ---")

    print("--- population ---")
    population = [solver.boundary_handler.transform(x) for x in solver.pop_sorted]
    fitnesses = solver.fit.fit

    pop_params = defaultdict(list)
    for solution_vec in population:
        params = objective.vector_to_params(list(solution_vec))
        for k, v in params.items():
            pop_params[k].append(v)
    pop_params["error"] = [float(f) for f in fitnesses]

    for name, values in pop_params.items():
        print(f"  {name}: {values}")

    # --- Best-so-far data ---
    if solver.best.x is None:
        return

    print("--- best_so_far ---")
    best_vec = list(solver.best.x)
    best_error = float(solver.best.f)
    best_params = objective.vector_to_params(best_vec)

    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"error: {best_error}\n")


def main():
    import time

    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Zeroth-order optimisation of simulation parameters using CMA-ES."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rope_uiuc",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/cfg/cmaes.yaml",
        help="Hydra base configuration YAML file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to *processed* dataset root directory.",
    )
    parser.add_argument(
        "--source_dataset_root",
        type=str,
        default=None,
        help="Path to *raw* source dataset directory (pre-processed tele-operation data).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=20,
        help="Maximum CMA-ES iterations (generations).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=8,
        help="Number of random dataset samples to evaluate per objective function call.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--cfg_options",
        nargs="*",
        default=[],
        help="Extra Hydra-style config overrides, e.g. --cfg-options sim.dt=0.033333333333 train.dataset_skip_frame=1",
    )
    parser.add_argument(
        "--num_steps_train",
        type=int,
        default=50,
        help="Number of time steps to train on.",
    )
    parser.add_argument(
        "--chamfer_weight",
        type=float,
        default=0.0,
        help="Weight for Chamfer distance term in loss.",
    )
    parser.add_argument(
        "--track_weight",
        type=float,
        default=1.0,
        help="Weight for tracking term (MSE) in loss.",
    )
    parser.add_argument(
        "--disable_graph",
        action="store_true",
        help="Disable Warp CUDA graph optimisation (enabled by default).",
    )

    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = f"log/{args.name}/dataset-dx0.02-s1.0/state"
    if args.source_dataset_root is None:
        args.source_dataset_root = (
            f"data/meta-material/data/{args.name}_merged/sub_episodes_v"
        )

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    set_all_seeds(args.seed)

    cfg_overrides = resolve_cfg_overrides(args.name, args.cfg_options)
    cfg = build_cfg(args.config, overrides=cfg_overrides)
    cfg.sim.num_steps_train = args.num_steps_train
    param_bounds = get_param_bounds(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(
        cfg, Path(args.dataset_root), Path(args.source_dataset_root), device=device
    )

    # Get initial parameter vector (normalised) from current cfg
    init_params = {
        "damping": float(cfg.sim.damping),
        "stiffness": float(cfg.sim.stiffness),
        "threshold": float(cfg.sim.threshold),
        "max_springs_per_node": int(cfg.sim.max_springs_per_node),
        "ground_friction": float(cfg.sim.ground_friction),
    }
    objective = SimulationObjective(
        cfg,
        dataset,
        str(device),
        param_bounds=param_bounds,
        sample_size=args.sample_size,
        seed=args.seed,
        use_graph=not args.disable_graph,
        chamfer_weight=args.chamfer_weight,
        track_weight=args.track_weight,
    )
    x_init = objective.to_search_vector(init_params)

    # CMA-ES configuration
    sigma0 = 0.20  # Initial std-dev in the normalised search space
    es = cma.CMAEvolutionStrategy(
        x_init, sigma0, {"bounds": [0.0, 1.0], "seed": args.seed}
    )

    # --- Evaluate initial parameters before starting optimisation ---
    print("\n--- CMA-ES Iteration 0 ---")
    initial_loss = objective(x_init)
    print("--- best_so_far ---")
    for k, v in init_params.items():
        print(f"{k}: {v}")
    print(f"error: {initial_loss}")

    # Start optimisation while invoking the callback every generation.
    es.optimize(
        objective,
        iterations=args.max_iter,
        callback=lambda solver: _log_params_callback(solver, objective),
        # n_jobs=es.popsize,
        n_jobs=2,
    )

    # Fetch results and convert back to parameter dictionary
    res = es.result  # (best_parameters, best_f, ...)  – see pycma docs
    best_vec: List[float] = list(res[0])
    best_error: float = float(res[1])
    best_params = objective.vector_to_params(best_vec)

    print("\n================ CMA-ES optimisation complete ================")
    print("Best normalised vector :", best_vec)
    print("Best error             :", best_error)
    print("Best parameters        :")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Optionally: write best parameters to disk as JSON for later reuse.
    out_path = Path("cmaes_optim/outputs/best_sim_params.json")
    import json

    with out_path.open("w") as f:
        json.dump({**best_params, "error": best_error}, f, indent=2)
    print(f"\nSaved best parameters to {out_path.resolve()}")

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60} minutes")


if __name__ == "__main__":
    main()
