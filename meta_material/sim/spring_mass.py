import numpy as np
import torch
import torch.nn as nn
import warp as wp
from warp.sim.render import SimRenderer
from scipy.spatial import cKDTree
import random

torch.autograd.set_detect_anomaly(True)


@wp.kernel
def add_springs_kernel(
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # --- new springs
    new_indices_i: wp.array(dtype=int),
    new_indices_j: wp.array(dtype=int),
    new_rest_lengths: wp.array(dtype=float),
    new_stiffness: wp.array(dtype=float),
    new_damping: wp.array(dtype=float),
    offset: int,
):
    tid = wp.tid()

    i = new_indices_i[tid]
    j = new_indices_j[tid]

    spring_indices[offset + tid * 2] = i
    spring_indices[offset + tid * 2 + 1] = j
    spring_rest_length[offset + tid] = new_rest_lengths[tid]
    spring_stiffness[offset + tid] = new_stiffness[tid]
    spring_damping[offset + tid] = new_damping[tid]


@wp.kernel
def update_springs(
    p_q: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_k: wp.array(dtype=float),
    spring_d: wp.array(dtype=float),
    spring_rest: wp.array(dtype=float),
    threshold: float,
    full_k: float,
    full_d: float,
):
    idx = wp.tid()
    i = spring_indices[idx * 2]
    j = spring_indices[idx * 2 + 1]

    pi = p_q[i]
    pj = p_q[j]
    dist = wp.length(pi - pj)

    if wp.length(pi - pj) <= threshold:
        spring_k[idx] = full_k
        spring_d[idx] = full_d
    else:
        spring_k[idx] = 0.0
        spring_d[idx] = 0.0

    spring_rest[idx] = dist


@wp.kernel
def apply_gripper_forces(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    base_particle_inv_mass: wp.array(dtype=float),
    gripper_centers: wp.array(dtype=wp.vec3, ndim=2),
    gripper_vels: wp.array(dtype=wp.vec3, ndim=2),
    gripper_closed: wp.array(dtype=float, ndim=2),
    gripper_radius: float,
    batch_size: int,
    points_per_env: int,
    teddy_mode: float,
    poke_stiffness: float,
    dt: float,
):
    """Apply gripper forcing to particles within gripper radius."""
    b, p = wp.tid()

    if b >= batch_size or p >= points_per_env:
        return

    if gripper_centers.shape[1] == 0:
        return

    particle_index = b * points_per_env + p
    particle_pos = particle_q[particle_index]

    # Use a numeric flag compatible with Warp's codegen; avoid Python booleans/breaks
    locked_flag = float(0.0)

    for gripper_id in range(gripper_centers.shape[1]):
        gripper_center = gripper_centers[b, gripper_id]
        gripper_vel = gripper_vels[b, gripper_id]
        is_closed = gripper_closed[b, gripper_id]

        if teddy_mode > 0.5:
            # Teddy dataset: apply a repulsive poke-like impulse for points
            # inside the gripper sphere instead of rigidly attaching them.
            particle_from_gripper = particle_pos - gripper_center
            distance = wp.length(particle_from_gripper)
            if distance < gripper_radius:
                direction = wp.normalize(particle_from_gripper + wp.vec3(1e-9, 1e-9, 1e-9))
                penetration = gripper_radius - distance
                inv_mass = base_particle_inv_mass[particle_index]
                particle_qd[particle_index] = (
                    particle_qd[particle_index]
                    + direction * poke_stiffness * penetration * inv_mass * dt
                )
        else:
            particle_from_gripper = particle_pos - gripper_center
            distance = wp.length(particle_from_gripper)

            if distance < gripper_radius and is_closed > 0.5:
                particle_qd[particle_index] = gripper_vel
                locked_flag = float(1.0)

    if teddy_mode > 0.5:
        particle_inv_mass[particle_index] = base_particle_inv_mass[particle_index]
    else:
        if locked_flag > 0.5:
            # Make particle kinematic by setting inverse mass to 0
            particle_inv_mass[particle_index] = 0.0
        else:
            # Restore original inverse mass
            particle_inv_mass[particle_index] = base_particle_inv_mass[particle_index]


@wp.kernel
def apply_gripper_forces_flag_dataset(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    base_particle_inv_mass: wp.array(dtype=float),
    gripper_vels: wp.array(dtype=wp.vec3, ndim=2),
    gripper_closed: wp.array(dtype=float, ndim=2),
    flag_y_thresholds: wp.array(dtype=float),
    batch_size: int,
    points_per_env: int,
):
    """Flag behavior: lock particles in top band while any gripper is closed."""
    b, p = wp.tid()

    if b >= batch_size or p >= points_per_env:
        return

    if gripper_vels.shape[1] == 0:
        return

    particle_index = b * points_per_env + p
    particle_pos = particle_q[particle_index]
    locked_flag = float(0.0)

    # Mirror reference flag logic: points in top band move with closed gripper.
    if particle_pos[1] > flag_y_thresholds[b]:
        for gripper_id in range(gripper_vels.shape[1]):
            is_closed = gripper_closed[b, gripper_id]
            if is_closed > 0.5:
                particle_qd[particle_index] = gripper_vels[b, gripper_id]
                locked_flag = float(1.0)

    if locked_flag > 0.5:
        particle_inv_mass[particle_index] = 0.0
    else:
        particle_inv_mass[particle_index] = base_particle_inv_mass[particle_index]


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        wp.set_seed(seed)
    except Exception:
        pass


class ImplicitBatchSim(nn.Module):
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        grippers: np.ndarray,
        points_per_env: int,
        batch_size: int,
        threshold: float = 0.02,
        stiffness: float = 1000.0,
        damping: float = 10.0,
        mass_per_point: float = 1.0,
        device=None,
        env_dist: int = 1,
        visualize: bool = False,
        sim_dt: float = 1.0 / 60.0,
        sim_substeps: int = 500,
        seed: int = 42,
        gripper_radius: float = 0.04,
        use_flag_dataset_behavior: bool = False,
        is_teddy_dataset: bool = False,
        poke_stiffness: float = 100.0,
        max_springs_per_node: int = 10,
        record_trajectory: bool = False,
        ground_friction: float = 0.5,
    ) -> None:
        set_seed(seed)
        super().__init__()
        self.x = x
        self.points_per_env = points_per_env
        # Always infer the true batch size from the provided data to avoid
        # shape/view mismatches when the caller passes an incorrect value.
        self.batch_size = len(x)
        self.threshold = threshold
        self.stiffness = stiffness
        self.damping = damping
        self.env_dist = env_dist
        self.sim_dt = sim_dt
        self.sim_substeps = sim_substeps
        self.gripper_radius = gripper_radius
        self.use_flag_dataset_behavior = use_flag_dataset_behavior
        self.is_teddy_dataset = is_teddy_dataset
        self.poke_stiffness = poke_stiffness
        self.max_springs_per_node = max_springs_per_node
        self.record_trajectory = record_trajectory
        self.ground_friction = ground_friction
        self._traj_points = []
        self._traj_grippers = []

        sim_device = device or wp.get_device()

        builder = wp.sim.ModelBuilder()
        self.build_model(
            builder,
            x,
            v,
            threshold,
            stiffness,
            damping,
            mass_per_point,
            ground_friction,
            device=sim_device,
        )

        self.model = builder.finalize(device=sim_device, requires_grad=False)
        self.model.ground = False
        self.model.threshold = threshold
        self.model.stiffness = stiffness
        self.model.damping = damping
        self.model.batch_size = self.batch_size
        self.model.points_per_env = self.total_points_per_env
        self.model.spring_count = int(self.model.spring_indices.shape[0] // 2)
        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator()
        self.model.sim_substeps = self.sim_substeps
        self.model.dt = self.sim_dt / self.model.sim_substeps
        wp.sim.collide(self.model, self.model.state())

        # Keep a persistent copy of the original inverse masses so we can
        # zero them for gripper-locked particles and restore afterward.
        base_inv_mass_torch = wp.to_torch(self.model.particle_inv_mass).clone()
        self._base_particle_inv_mass = wp.from_torch(
            base_inv_mass_torch, dtype=wp.float32
        )

        self.visualize = visualize
        self.current_step = 0
        if self.visualize:
            self.renderer = SimRenderer(
                self.model, "implicit_batch_sim.usd", scaling=100.0
            )
        else:
            self.renderer = None

        # ------------------------------------------------------------------
        #  Persistent simulation buffers & optional CUDA graph setup
        # ------------------------------------------------------------------
        # Create two states that will be reused every forward() call. Re-using
        # the same Warp `State` objects is a prerequisite for CUDA Graph
        # capture as it guarantees that the graph kernel launch topology is
        # identical across launches.

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Flags / placeholders required for CUDA graph support.
        self.use_cuda_graph = wp.get_device().is_cuda
        self._graph_no_gripper = None  # Graph when *no* gripper forces are used
        self._graph_with_gripper = None  # Graph when gripper forces are active

        # These arrays will hold the per-environment gripper data when present.
        self._gripper_centers = None
        self._gripper_vels = None
        self._gripper_closed = None
        try:
            if self.use_flag_dataset_behavior:
                y_thresh_np = np.array(
                    [
                        float(np.max(x[e][:, 1]) - float(self.gripper_radius))
                        for e in range(len(x))
                    ],
                    dtype=np.float32,
                )
            else:
                y_thresh_np = np.zeros(len(x), dtype=np.float32)
        except Exception:
            y_thresh_np = np.zeros(len(x), dtype=np.float32)
        self._flag_y_thresholds = wp.array(
            y_thresh_np, dtype=wp.float32, device=sim_device
        )

    def get_initial_state(self):
        """
        Returns the full initial state (positions and velocities) of all particles,
        including newly added internal ones. Offsets are removed from positions.
        """
        if not hasattr(self, "model"):
            return None, None

        x0 = wp.to_torch(self.model.particle_q).view(
            self.batch_size, self.model.points_per_env, 3
        )
        v0 = wp.to_torch(self.model.particle_qd).view(
            self.batch_size, self.model.points_per_env, 3
        )

        # The positions in the model have the env_dist offset. We need to remove it.
        offsets = torch.zeros_like(x0)
        offsets[..., 0] = (
            torch.arange(self.batch_size, device=x0.device, dtype=x0.dtype)
            .mul(self.env_dist)
            .unsqueeze(1)
        )
        x0_no_offset = x0 - offsets
        return x0_no_offset, v0

    def build_model(
        self,
        builder,
        base,
        v,
        threshold=0.1,
        stiffness=100.0,
        damping=0.0,
        mass_per_point=1.0,
        ground_friction: float = 0.5,
        device=None,
    ):
        batch_size = len(base)
        env_slices = []
        all_env_pts = []
        self.total_points_per_env = 0

        for e in range(batch_size):
            start_idx = builder.particle_count
            pts = base[e]

            # If any point is below ground_y, bring it to lowest_y
            lowest_y = 0.04
            pts[pts[:, 1] < lowest_y - 1e-5, 1] = lowest_y

            all_env_pts.append(pts)

            offset = np.array([e * self.env_dist, 0.0, 0.0], dtype=np.float32)

            for i in range(len(pts)):
                builder.add_particle(
                    pos=(pts[i] + offset).tolist(),
                    vel=v[e][i].tolist(),
                    mass=mass_per_point,
                    radius=0.001,
                )

            env_slices.append((start_idx, builder.particle_count))
            if e == 0:
                self.total_points_per_env = builder.particle_count - start_idx

        spring_indices_i = []
        spring_indices_j = []
        spring_rest_lengths = []
        spring_stiffnesses = []
        spring_dampings = []

        for e, (start_idx, end_idx) in enumerate(env_slices):
            pts = all_env_pts[e]
            if len(pts) <= 1:
                continue

            tree = cKDTree(pts)

            # Query only a handful of nearest neighbours for each point, instead of *all*
            k_query = (
                self.max_springs_per_node + 3
            )  # small cushion to reduce risk of missing edges
            # ``tree.query`` returns (dists, indices) arrays of shape (N, k_query+1)
            # The first neighbour is always the point itself (dist==0).
            dists_mat, nbrs_mat = tree.query(
                pts,
                k=k_query + 1,  # include the point itself
                distance_upper_bound=threshold,
                workers=-1,  # multithreaded C implementation inside SciPy
            )

            candidate_pairs: list[tuple[int, int]] = []
            candidate_dists: list[float] = []

            n_pts = len(pts)
            for i_local in range(n_pts):
                row_nbrs = nbrs_mat[i_local]
                row_dists = dists_mat[i_local]
                for j_local, dist in zip(row_nbrs, row_dists):
                    # Skip self, invalid, or out-of-threshold neighbours
                    if j_local == i_local or j_local >= n_pts or np.isinf(dist):
                        continue
                    if (not np.isfinite(dist)) or dist <= 1e-9:
                        continue
                    # Enforce an ordering to avoid duplicates
                    if i_local < j_local:
                        candidate_pairs.append((i_local, j_local))
                        candidate_dists.append(dist)

            if len(candidate_pairs) > 0:
                pairs = np.asarray(candidate_pairs, dtype=np.int32)
                dists = np.asarray(candidate_dists, dtype=pts.dtype)

                # Sort all candidate pairs (within the threshold) by their length
                sorted_idx = np.argsort(dists)

                # Running count of how many springs are connected to each particle
                spring_counts = np.zeros(n_pts, dtype=int)

                for idx in sorted_idx:
                    i_local, j_local = pairs[idx]
                    dist = dists[idx]

                    if dist <= 1e-9:
                        continue

                    # Skip if either particle already reached its spring budget
                    if spring_counts[i_local] >= self.max_springs_per_node:
                        continue
                    if spring_counts[j_local] >= self.max_springs_per_node:
                        continue

                    # Global particle indices in the batched builder
                    global_i = start_idx + i_local
                    global_j = start_idx + j_local

                    rest_len = float(dist)
                    spring_indices_i.append(global_i)
                    spring_indices_j.append(global_j)
                    spring_rest_lengths.append(rest_len)
                    spring_stiffnesses.append(stiffness)
                    spring_dampings.append(damping)

                    spring_counts[i_local] += 1
                    spring_counts[j_local] += 1

        num_new_springs = len(spring_indices_i)
        if num_new_springs > 0:
            new_indices_flat = []
            for i in range(num_new_springs):
                new_indices_flat.append(spring_indices_i[i])
                new_indices_flat.append(spring_indices_j[i])

            builder.spring_indices.extend(new_indices_flat)
            builder.spring_rest_length.extend(spring_rest_lengths)
            builder.spring_stiffness.extend(spring_stiffnesses)
            builder.spring_damping.extend(spring_dampings)

        # lowest_y = (
        #     float(base[..., 1].min())
        #     if isinstance(base, np.ndarray)
        #     else float(base.min(axis=(0, 1))[1])
        # )
        lowest_y = 0.04
        self.ground_y = lowest_y - 1e-5

        box_half_height = 0.05
        builder.add_shape_box(
            body=-1,
            pos=(0.0, self.ground_y - box_half_height, 0.0),
            hx=100.0,
            hy=box_half_height,
            hz=100.0,
            is_solid=True,
            mu=ground_friction,
        )

    def render(self, state):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(state)
            self.renderer.end_frame()

    def save_renderer(self):
        """Save the renderer after all steps are complete"""
        if self.renderer is not None:
            self.renderer.save()

    def forward(
        self,
        step: int,
        x: torch.Tensor,
        v: torch.Tensor,
        point_feats,
        grippers: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the Warp simulation by one outer-loop step.

        The previous implementation delegated this work to the custom
        autograd.Function ``ImplicitNewBatchSimFunction`` so that
        gradients could be propagated through the simulation.  We have
        determined that gradient flow through the simulator is not
        required, so the logic is inlined here directly with gradient
        tracking disabled.
        """

        use_grippers = grippers is not None and grippers.shape[2] > 0

        self.current_step = step

        B, N, _ = x.shape
        device = x.device

        # Offset each environment along the x-axis so that the batched
        # simulations do not intersect.
        offsets = torch.zeros_like(x)
        offsets[..., 0] = (
            torch.arange(B, device=device, dtype=x.dtype)
            .mul(self.env_dist)
            .unsqueeze(1)
        )

        x_offset = x + offsets

        # ------------------------------------------------------------------ #
        #  Write the *current* positions/velocities into the persistent state
        # ------------------------------------------------------------------ #

        x_flat = x_offset.reshape(-1, 3).contiguous()
        v_flat = v.reshape(-1, 3).contiguous()

        self.state_0.particle_q.assign(
            wp.from_torch(x_flat, dtype=wp.vec3, requires_grad=False)
        )
        self.state_0.particle_qd.assign(
            wp.from_torch(v_flat, dtype=wp.vec3, requires_grad=False)
        )

        wp.sim.collide(self.model, self.state_0)

        # ------------------------------------------------------------------ #
        #  Handle gripper data (if supplied)
        # ------------------------------------------------------------------ #

        if use_grippers:
            gripper_xyz = grippers[:, step, :, :3].clone()
            gripper_v = grippers[:, step, :, 3:6].clone()
            gripper_closed = grippers[:, step, :, -1] < 0.5

            g_offsets = torch.zeros_like(gripper_xyz)
            g_offsets[..., 0] = (
                torch.arange(B, device=gripper_xyz.device, dtype=gripper_xyz.dtype)
                .mul(self.env_dist)
                .unsqueeze(1)
            )
            gripper_xyz_offset = gripper_xyz + g_offsets

            # Lazily allocate Warp arrays for gripper data (once)
            if self._gripper_centers is None:
                G = gripper_xyz.shape[1]
                self._gripper_centers = wp.empty(
                    (B, G), dtype=wp.vec3, device=self.model.device
                )
                self._gripper_vels = wp.empty(
                    (B, G), dtype=wp.vec3, device=self.model.device
                )
                self._gripper_closed = wp.empty(
                    (B, G), dtype=wp.float32, device=self.model.device
                )

            # Copy current values
            self._gripper_centers.assign(
                wp.from_torch(gripper_xyz_offset.contiguous(), dtype=wp.vec3)
            )
            self._gripper_vels.assign(
                wp.from_torch(gripper_v.contiguous(), dtype=wp.vec3)
            )
            self._gripper_closed.assign(
                wp.from_torch(gripper_closed.float().contiguous(), dtype=wp.float32)
            )

        # ------------------------------------------------------------------ #
        #  Run the internal Warp simulation – either through a CUDA graph or
        #  via the plain Python loop depending on availability.
        # ------------------------------------------------------------------ #

        if self.use_cuda_graph:
            graph_attr = "_graph_with_gripper" if use_grippers else "_graph_no_gripper"

            if getattr(self, graph_attr) is None:
                # First time we encounter this branch – capture the graph.
                with wp.ScopedCapture() as capture:
                    self._simulate_substeps(use_grippers)
                setattr(self, graph_attr, capture.graph)

            wp.capture_launch(getattr(self, graph_attr))
        else:
            # Fallback: run regular Python loop (still on GPU).
            self._simulate_substeps(use_grippers)

        if self.renderer is not None:
            self.renderer.begin_frame(self.current_step)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

        # ------------------------------------------------------------------ #
        #  Fetch results back to PyTorch
        # ------------------------------------------------------------------ #

        final_state = self.state_0  # After _simulate_substeps(), state_0 is current

        x_flat_out = wp.to_torch(final_state.particle_q)
        v_flat_out = wp.to_torch(final_state.particle_qd)

        x_out = x_flat_out.view(self.batch_size, self.model.points_per_env, 3)
        v_out = v_flat_out.view(self.batch_size, self.model.points_per_env, 3)

        x_out = x_out - offsets

        # --- RECORD TRAJECTORY ---
        if self.record_trajectory:
            # Save as numpy arrays for easier serialization
            self._traj_points.append(x_out.detach().cpu().numpy())
            if grippers is not None:
                # Save gripper positions for this step (B, G, 7) or (B, G, 6)
                if grippers.shape[1] > step:
                    self._traj_grippers.append(grippers[:, step].detach().cpu().numpy())
                else:
                    self._traj_grippers.append(None)
            else:
                self._traj_grippers.append(None)
        # --- END RECORD ---

        return x_out, v_out

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _simulate_substeps(self, use_grippers: bool):
        """Run *sim_substeps* internal Warp steps starting from *self.state_0*.

        The function operates **in-place** on ``self.state_0`` / ``self.state_1``.
        It is intentionally free of any Python branching that would change the
        launch topology between calls, making it suitable for CUDA graph
        capture.  Gripper forces are only applied when *use_grippers* is True.
        """

        # # If no grippers, ensure all particles are fully dynamic (restore masses)
        # if not use_grippers:
        #     self.model.particle_inv_mass.assign(self._base_particle_inv_mass)

        for _ in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)

            self.state_0.clear_forces()

            if use_grippers:
                if self.use_flag_dataset_behavior:
                    wp.launch(
                        apply_gripper_forces_flag_dataset,
                        dim=(self.batch_size, self.model.points_per_env),
                        inputs=[
                            self.state_0.particle_q,
                            self.state_0.particle_qd,
                            self.model.particle_inv_mass,
                            self._base_particle_inv_mass,
                            self._gripper_vels,
                            self._gripper_closed,
                            self._flag_y_thresholds,
                            self.batch_size,
                            self.model.points_per_env,
                        ],
                        device=self.model.device,
                    )
                else:
                    wp.launch(
                        apply_gripper_forces,
                        dim=(self.batch_size, self.model.points_per_env),
                        inputs=[
                            self.state_0.particle_q,
                            self.state_0.particle_qd,
                            self.model.particle_inv_mass,
                            self._base_particle_inv_mass,
                            self._gripper_centers,
                            self._gripper_vels,
                            self._gripper_closed,
                            self.gripper_radius,
                            self.batch_size,
                            self.model.points_per_env,
                            float(1.0 if self.is_teddy_dataset else 0.0),
                            float(self.poke_stiffness),
                            float(self.model.dt),
                        ],
                        device=self.model.device,
                    )

            # Semi-implicit Euler integration
            self.integrator.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.model.dt,
            )

            # Swap state references (ping-pong buffer)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def save_trajectory(self, out_path):
        """
        Save the recorded trajectory to a .npz file. Includes point positions and gripper positions.
        """
        if not self.record_trajectory:
            raise RuntimeError("Trajectory recording was not enabled.")
        np.savez_compressed(
            out_path,
            points=np.array(self._traj_points),  # (T, B, N, 3)
            grippers=np.array(self._traj_grippers, dtype=object),  # (T, B, G, ...)
        )
