import torch
import einops
import numpy as np
import logging
import os
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
import time
from omni_drones.utils.torch import quat_rotate_inverse
import importlib



class NavigationEnv(IsaacEnv):
    
    def __init__(self, cfg):
        print("[Navigation Environment]: Init Environment...")
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        self.vertical_ray_angles_deg = torch.cat(
            [
                torch.tensor([-90.0]),
                torch.linspace(*self.lidar_vfov, self.lidar_vbeams),
                torch.tensor([90.0]),
            ]
        )
        self.lidar_vbeams_ext = self.lidar_vbeams + 2


        map_name = cfg.env.map_name 
        obstacle_module_name = f"obs_{map_name}"
        print(f"[Navigation Environment]: Loading Map: {obstacle_module_name}.py")
        
        try:
            obs_module = importlib.import_module(obstacle_module_name)
        except ImportError as e:
            logging.error(f"Error: Can not load Map '{obstacle_module_name}'. Back to 'obs_vanilla'. Error: {e}")
            import obs_vanilla as obs_module 
        
        self.spawn_static_obstacles = obs_module.spawn_static_obstacles
        self.DynamicObstacleManager = obs_module.DynamicObstacleManager


        super().__init__(cfg, cfg.headless)

        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        

        self.curriculum_step = 0
        self._last_training_flag = bool(self.training)
        self._curriculum_step_checkpoint = 0

        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,
                vertical_ray_angles=self.vertical_ray_angles_deg
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams_ext)
        
        # ==================== whole-body shape scan ====================
        self.shape_scan = self._compute_shape_scan(cfg.drone.marker.shape)
        print(f"shape_scan: {self.shape_scan.shape}")
        # ==================== whole-body shape scan ====================

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)

    # ==================== whole-body ====================
    def _compute_shape_scan(self, shape_name):

        obj_folder = os.path.join(os.path.dirname(__file__), "..", "obj")
        obj_file = os.path.join(obj_folder, f"{shape_name}.obj")
        
        if not os.path.exists(obj_file):
            print(f"No OBJ file: {obj_file}")
            return torch.zeros(1, self.lidar_hbeams, self.lidar_vbeams_ext, device=self.device)

        vertices = []
        faces = []
        with open(obj_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    faces.append(face)
        
        if not vertices or not faces:
            print(f"Invaild OBJ File: {obj_file}")
            return torch.zeros(1, self.lidar_hbeams, self.lidar_vbeams_ext, device=self.device)
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        horizontal_angles = np.linspace(0, 2 * np.pi, self.lidar_hbeams, endpoint=False)
        vertical_angles = np.deg2rad(self.vertical_ray_angles_deg.cpu().numpy())
        shape_distances = np.zeros((self.lidar_hbeams, self.lidar_vbeams_ext))
        
        for h_idx, h_angle in enumerate(horizontal_angles):
            for v_idx, v_angle in enumerate(vertical_angles):
                ray_dir = np.array([
                    np.cos(v_angle) * np.cos(h_angle),
                    np.cos(v_angle) * np.sin(h_angle),
                    np.sin(v_angle)                     
                ])
                
                max_distance = self._ray_mesh_intersection(
                    ray_origin=np.array([0.0, 0.0, 0.0]),
                    ray_direction=ray_dir,
                    vertices=vertices,
                    faces=faces
                )
                shape_distances[h_idx, v_idx] = max_distance
        
        shape_scan = torch.tensor(shape_distances, dtype=torch.float32, device=self.device)
        shape_scan = shape_scan.unsqueeze(0)
        
        print(f"Shape Scan Statistics - Min: {shape_scan.min():.4f}, Max: {shape_scan.max():.4f}, Avg: {shape_scan.mean():.4f}")
        
        save_folder = os.path.join(os.path.dirname(__file__), "..", "shape_scan_data")
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"{shape_name}_shape_scan.npz")
        
        np.savez(
            save_file,
            shape_scan=shape_scan.cpu().numpy(),
            shape_name=shape_name,
            lidar_hbeams=self.lidar_hbeams,
            lidar_vbeams=self.lidar_vbeams_ext,
            lidar_vfov=(-90.0, 90.0),
            horizontal_angles=horizontal_angles,
            vertical_angles=vertical_angles,
        )
        print(f"Shape Scan Saved in: {save_file}")
        
        return shape_scan
    
    def _ray_mesh_intersection(self, ray_origin, ray_direction, vertices, faces):
        epsilon = 1e-6
        max_distance = 0.0
        
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + epsilon)
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            if abs(a) < epsilon:
                continue
            
            f = 1.0 / a
            s = ray_origin - v0
            u = f * np.dot(s, h)
            
            if u < 0.0 or u > 1.0:
                continue
            
            q = np.cross(s, edge1)
            v = f * np.dot(ray_direction, q)
            
            if v < 0.0 or u + v > 1.0:
                continue
            

            t = f * np.dot(edge2, q)
        
            if t > epsilon:
                max_distance = max(max_distance, t)
        
        return max_distance
    # ==================== whole-body ====================

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        cfg = drone_model.cfg_cls(force_sensor=False)
        marker_cfg = getattr(self.cfg.drone, "marker", None)

        # ==================== whole-body visualization marker ====================
        if marker_cfg is not None and bool(marker_cfg.enable):
                cfg.center_marker = True
                if hasattr(marker_cfg, "shape"):  
                    cfg.center_marker_shape = str(marker_cfg.shape)
                if hasattr(marker_cfg, "color"):
                    cfg.center_marker_color = tuple(float(v) for v in marker_cfg.color)
                
                print(f"Enable Whole Body Visualization, Using OBJ File: {cfg.center_marker_shape}.obj")
        else:
            cfg.center_marker = False
        # ====================================================================
        
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]


        self.spawn_static_obstacles(self.cfg, self.num_envs, self.map_range)


        self.dyn_obs_manager = self.DynamicObstacleManager(self.cfg, self.map_range, self.device)


    # ==================== whole-body ====================
    def _set_specs(self):
        observation_dim = 7
        num_dim_each_dyn_obs_state = 9

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams_ext), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                    # "shape_scan": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams_ext), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec,
            })
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
    # ==================== whole-body ====================

    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights
            target_pos = target_pos * selected_masks + selected_shifts

            self.target_pos[env_ids] = target_pos
   
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights
            pos = pos * selected_masks + selected_shifts
            
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.

        # ==================== whole-body ====================
        min_flight_height = getattr(self.cfg.env, "min_flight_height", 0.5)
        max_flight_height = getattr(self.cfg.env, "max_flight_height", 4.0)
        self.height_range[env_ids, 0, 0] = min_flight_height
        self.height_range[env_ids, 0, 1] = max_flight_height
        # ==================== whole-body ====================

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        freeze_in_eval = bool(getattr(self.cfg.env, "curriculum_freeze_in_eval", True))
        if self.training != self._last_training_flag:
            if freeze_in_eval:
                if not self.training:
                    self._curriculum_step_checkpoint = self.curriculum_step
                else:
                    self.curriculum_step = self._curriculum_step_checkpoint
            self._last_training_flag = bool(self.training)

        if self.training:
            self.curriculum_step += 1
        
        self.dyn_obs_manager.update()

        self.lidar.update(self.dt)
    
    # ==================== whole-body ====================
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = self.root_state[..., :13]

        # ==================== whole-body scan ====================
        ray_distances = (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )
        clearance = (ray_distances - self.shape_scan).clamp(min=0.0)
        self.lidar_scan = clearance
        # ============================================================

        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            v_flat = v.reshape(-1, 3)
            origins = x.expand(v_flat.shape[0], 3)
            valid_mask = torch.isfinite(v_flat).all(dim=-1)
            for i in range(v_flat.shape[0]):
                if not valid_mask[i]:
                    continue
                self.debug_draw.vector(origins[i], v_flat[i])

        quat_w = self.root_state[..., 3:7]

        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True)
        

        target_dir_3d = self.target_dir.clone()

        rpos_clipped = rpos / distance.clamp(1e-6)
        rpos_clipped_b = quat_rotate_inverse(quat_w, rpos_clipped)
        
        vel_w = self.root_state[..., 7:10]
        vel_b = quat_rotate_inverse(quat_w, vel_w) 

        drone_state = torch.cat([rpos_clipped_b, distance, vel_b], dim=-1).squeeze(1)


        if (self.dyn_obs_manager.num_obstacles > 0):
            dyn_obs_pos_expanded = self.dyn_obs_manager.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_distance = torch.norm(dyn_obs_rpos_expanded, dim=2)
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False)
            dyn_obs_range_mask = dyn_obs_distance.gather(1, closest_dyn_obs_idx) > self.lidar_range

            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_3d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0.
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            closest_dyn_obs_size = self.dyn_obs_manager.dyn_obs_size[closest_dyn_obs_idx]

            # ==================== whole-body ====================
            dir_vec = closest_dyn_obs_rpos / closest_dyn_obs_distance.clamp_min(1e-6)
            phi = torch.atan2(dir_vec[..., 1], dir_vec[..., 0])
            two_pi = torch.tensor(2.0 * np.pi, device=self.device)
            phi = torch.remainder(phi + two_pi, two_pi)

            H = self.lidar_hbeams
            V = self.lidar_vbeams_ext

            v_angle = torch.atan2(dir_vec[..., 2], torch.norm(dir_vec[..., :2], dim=-1))
            deg = v_angle * (180.0 / np.pi)
            vfov_min = -90.0
            vfov_max = 90.0

            h_idx = torch.round(phi / two_pi * (H)).long() % H
            v_idx = torch.round((deg - vfov_min) / max(vfov_max - vfov_min, 1e-6) * (V - 1)).clamp(0, V - 1).long()

            shape_map = self.shape_scan[0].reshape(H * V)
            lin_idx = (h_idx * V + v_idx).view(-1)
            shape_r = shape_map[lin_idx].view(h_idx.shape)
            shape_r = shape_r.unsqueeze(-1)

            width_half = (closest_dyn_obs_size[..., 0].unsqueeze(-1)) * 0.5
            clearance_3d = (closest_dyn_obs_distance - shape_r - width_half).clamp_min(0.0)
            closest_dyn_obs_clearance_reward = clearance_3d.squeeze(-1)
            closest_dyn_obs_clearance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            # ==================== whole-body ====================

            closest_dyn_obs_vel = self.dyn_obs_manager.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_3d) 

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1.
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)      
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 9, device=self.cfg.device)
            
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_3d,
            "dynamic_obstacle": dyn_obs_states,
            # "shape_scan": self.shape_scan.expand(self.num_envs, -1, -1, -1)
        }


        safety_lambda = getattr(self.cfg.env, "safety_reward_lambda", 1.0)
        safety_k = getattr(self.cfg.env, "safety_reward_k", 0.5)
        clearance_squared = self.lidar_scan ** 2
        reward_safety_static = 1.0 - torch.exp(-safety_k * clearance_squared)
        reward_safety_static = reward_safety_static.mean(dim=(2, 3))


        if (self.dyn_obs_manager.num_obstacles > 0):
            clearance_dyn_squared = closest_dyn_obs_clearance_reward ** 2
            reward_safety_dynamic = safety_lambda * (1.0 - torch.exp(-safety_k * clearance_dyn_squared))
            reward_safety_dynamic = reward_safety_dynamic.mean(dim=-1, keepdim=True)


        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)

        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)


        # ==================== whole-body ====================
        drone_height = self.drone.pos[..., 2]
        if drone_height.dim() > 1:
            drone_height = drone_height.squeeze(-1)
        
        min_h = self.height_range[:, 0, 0]
        max_h = self.height_range[:, 0, 1]
        out_of_bounds_lower = torch.clamp(min_h - drone_height, min=0.0)
        out_of_bounds_upper = torch.clamp(drone_height - max_h, min=0.0)
        penalty_out_of_bounds = out_of_bounds_lower**2 + out_of_bounds_upper**2 
        height_penalty_weight = getattr(self.cfg.env, "height_penalty_weight", 4.0)
        reward_height = (- penalty_out_of_bounds * height_penalty_weight).unsqueeze(-1)
        # ==================== whole-body ====================


        # ==================== whole-body ====================
        lidar_min = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
        collision_threshold = 0.1
        static_collision = (lidar_min < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)

        if (self.dyn_obs_manager.num_obstacles > 0):
            dynamic_collision = (closest_dyn_obs_clearance_reward.min(dim=1, keepdim=True).values < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)
        else:
            dynamic_collision = torch.zeros_like(static_collision)
        # ==================== whole-body ====================
        
        collision = static_collision | dynamic_collision
        
        # ==================== whole-body ====================
        r1 = getattr(self.cfg.env, "curriculum_r1_steps", None)
        r2 = getattr(self.cfg.env, "curriculum_r2_steps", None)
        total_steps = float(getattr(self.cfg.env, "curriculum_total_steps", 1_000_000))
        step_divisor = float(getattr(self.cfg.env, "curriculum_step_divisor", 1.0))

        if self.training:
            if r1 is not None and r2 is not None and float(r2) > float(r1):
                step_now = float(self.curriculum_step)
                if step_now < float(r1):
                    p = 0.0
                elif step_now >= float(r2):
                    p = 1.0
                else:
                    p = (step_now - float(r1)) / (float(r2) - float(r1))
            else:
                effective_steps = self.curriculum_step / max(step_divisor, 1e-9)
                p = float(min(1.0, max(0.0, effective_steps / max(total_steps, 1e-9))))
        else:
            p = 1.0

        vel_w_start = float(getattr(self.cfg.env, "vel_weight_start", 1.5))
        vel_w_end = float(getattr(self.cfg.env, "vel_weight_end", 1.0))
        safety_w_start = float(getattr(self.cfg.env, "safety_weight_start", 0.8))
        safety_w_end = float(getattr(self.cfg.env, "safety_weight_end", 1.5))
        safety_dyn_w_start = float(getattr(self.cfg.env, "safety_dyn_weight_start", safety_w_start))
        safety_dyn_w_end = float(getattr(self.cfg.env, "safety_dyn_weight_end", safety_w_end))
        w_smoothe = float(getattr(self.cfg.env, "smoothness_weight", 0.1))
        r_bias = float(getattr(self.cfg.env, "reward_bias", 0.2))
        reach_goal_reward = float(getattr(self.cfg.env, "reach_goal_reward", 10.0))

        w_vel = vel_w_start + (vel_w_end - vel_w_start) * p
        w_safety_static = safety_w_start + (safety_w_end - safety_w_start) * p
        w_safety_dynamic = safety_dyn_w_start + (safety_dyn_w_end - safety_dyn_w_start) * p

        base_bias = r_bias

        log_interval = int(getattr(self.cfg.env, "curriculum_log_interval", 1000))
        should_log_train = self.training and (log_interval > 0) and (self.curriculum_step % log_interval == 0)
        should_log_eval = (not self.training) and (self.progress_buf[0].item() == 0)
        if should_log_train or should_log_eval:
            effective_step_display = (
                self.curriculum_step if self.training else (self._curriculum_step_checkpoint if log_interval >= 0 else self.curriculum_step)
            )
            if (self.dyn_obs_manager.num_obstacles > 0):
                logging.info(
                    f"[Curriculum] step={effective_step_display} p={p:.4f} "
                    f"w_vel={w_vel:.3f} w_safety_static={w_safety_static:.3f} "
                    f"w_safety_dynamic={w_safety_dynamic:.3f}"
                )
            else:
                logging.info(
                    f"[Curriculum] step={effective_step_display} p={p:.4f} "
                    f"w_vel={w_vel:.3f} w_safety={w_safety_static:.3f}"
                )

        if (self.dyn_obs_manager.num_obstacles > 0):
            self.reward = (
                reward_vel * w_vel
                + base_bias
                + reward_safety_static * w_safety_static
                + reward_safety_dynamic * w_safety_dynamic
                - penalty_smooth * w_smoothe
                + reward_height 
            )
        else:
            self.reward = (
                reward_vel * w_vel
                + base_bias
                + reward_safety_static * w_safety_static
                - penalty_smooth * w_smoothe
                + reward_height 
            )
        # ==================== whole-body ====================
        

        # 终止条件
        reach_goal = (distance.squeeze(-1) < 0.5)

        terminal_goal_reward = 10.0
        self.reward = self.reward + (reach_goal.float() * terminal_goal_reward)
        
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)
    # ==================== whole-body ====================

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
