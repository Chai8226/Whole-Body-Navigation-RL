import torch
import numpy as np
import os
import time
import logging

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.orbit.terrains import (
    TerrainImporterCfg,
    TerrainImporter,
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from pxr import UsdGeom, Vt

from utils import construct_input


def spawn_static_obstacles(cfg, num_envs, map_range):
    """
    在场景中生成静态地形和水平横梁。
    """
    print("[ObstacleManager]: Generating Static Obstacles (Terrain and Beams)...")
    terrain_cfg = TerrainImporterCfg(
        num_envs=num_envs,
        env_spacing=0.0,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(map_range[0] * 2, map_range[1] * 2),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            color_scheme="height",
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=cfg.env.num_obstacles,
                    obstacle_height_mode="range",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                    obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                    platform_width=0.0,
                ),
            },
        ),
        visual_material=None,
        max_init_terrain_level=None,
        collision_group=-1,
        debug_vis=True,
    )
    terrain_importer = TerrainImporter(terrain_cfg)

# ==================== whole-body (Modified for Static Shapes) ====================
    # 将静态障碍物添加到地形网格中，以便LiDAR可以检测到它们

    total_static_obstacles = int(getattr(cfg.env, "num_obstacles", 24))
    percent_static_cubes = float(getattr(cfg.env, "percent_cubes", 0.3))
    percent_static_cylinders = float(getattr(cfg.env, "percent_cylinders", 0.3))
    percent_static_spheres = float(getattr(cfg.env, "percent_spheres", 0.3))

    num_static_cubes = int(percent_static_cubes * total_static_obstacles)
    num_static_cylinders = int(percent_static_cylinders * total_static_obstacles)
    num_static_spheres = int(percent_static_spheres * total_static_obstacles)
    
    cube_size_range = tuple(getattr(cfg.env, "cube_size_range", [0.5, 2.0]))
    cylinder_height_range = tuple(getattr(cfg.env, "cylinder_height_range", [1.0, 6.0]))
    cylinder_radius_range = tuple(getattr(cfg.env, "cylinder_radius_range", [0.2, 0.8]))
    sphere_radius_range = tuple(getattr(cfg.env, "sphere_radius_range", [0.5, 1.5]))
    clearance_range = tuple(getattr(cfg.env, "clearance_range", [0.1, 1.5]))

    stage = prim_utils.get_current_stage()
    terrain_mesh_prim_path = "/World/ground/terrain/mesh"
    terrain_mesh_prim = UsdGeom.Mesh.Get(stage, terrain_mesh_prim_path)

    if terrain_mesh_prim:
        existing_points = list(terrain_mesh_prim.GetPointsAttr().Get())
        existing_face_counts = list(
            terrain_mesh_prim.GetFaceVertexCountsAttr().Get()
        )
        existing_face_indices = list(
            terrain_mesh_prim.GetFaceVertexIndicesAttr().Get()
        )
        vertex_offset = len(existing_points)

        obstacle_meshes = []
        total_added_obstacles = 0

        # 生成 立方体 (Cubes)
        for _ in range(num_static_cubes):
            L = float(np.random.uniform(*cube_size_range))
            clearance = float(np.random.uniform(*clearance_range))
            x = float(np.random.uniform(-map_range[0] + 1.5, map_range[0] - 1.5))
            y = float(np.random.uniform(-map_range[1] + 1.5, map_range[1] - 1.5))
            z = (L / 2.0) + clearance  # 放置在地面上 + 间隙

            lx, ly, lz = L / 2, L / 2, L / 2
            box_verts = np.array(
                [
                    [x - lx, y - ly, z - lz],
                    [x + lx, y - ly, z - lz],
                    [x + lx, y + ly, z - lz],
                    [x - lx, y + ly, z - lz],
                    [x - lx, y - ly, z + lz],
                    [x + lx, y - ly, z + lz],
                    [x + lx, y + ly, z + lz],
                    [x - lx, y + ly, z + lz],
                ]
            )
            obstacle_meshes.append((box_verts, vertex_offset))
            vertex_offset += 8
            total_added_obstacles += 1

        # 生成 圆柱体 (Cylinders) - 近似为长方体
        for _ in range(num_static_cylinders):
            R = float(np.random.uniform(*cylinder_radius_range))
            H = float(np.random.uniform(*cylinder_height_range))
            clearance = float(np.random.uniform(*clearance_range))
            x = float(np.random.uniform(-map_range[0] + 1.5, map_range[0] - 1.5))
            y = float(np.random.uniform(-map_range[1] + 1.5, map_range[1] - 1.5))
            z = (H / 2.0) + clearance  # 放置在地面上 + 间隙

            lx, ly, lz = R, R, H / 2  # 近似为 (2R x 2R x H) 的长方体
            box_verts = np.array(
                [
                    [x - lx, y - ly, z - lz],
                    [x + lx, y - ly, z - lz],
                    [x + lx, y + ly, z - lz],
                    [x - lx, y + ly, z - lz],
                    [x - lx, y - ly, z + lz],
                    [x + lx, y - ly, z + lz],
                    [x + lx, y + ly, z + lz],
                    [x - lx, y + ly, z + lz],
                ]
            )
            obstacle_meshes.append((box_verts, vertex_offset))
            vertex_offset += 8
            total_added_obstacles += 1

        # 生成 球体 (Spheres) - 近似为立方体
        for _ in range(num_static_spheres):
            R = float(np.random.uniform(*sphere_radius_range))
            clearance = float(np.random.uniform(*clearance_range))
            x = float(np.random.uniform(-map_range[0] + 1.5, map_range[0] - 1.5))
            y = float(np.random.uniform(-map_range[1] + 1.5, map_range[1] - 1.5))
            z = R + clearance  # 放置在地面上 + 间隙

            lx, ly, lz = R, R, R  # 近似为 (2R x 2R x 2R) 的立方体
            box_verts = np.array(
                [
                    [x - lx, y - ly, z - lz],
                    [x + lx, y - ly, z - lz],
                    [x + lx, y + ly, z - lz],
                    [x - lx, y + ly, z - lz],
                    [x - lx, y - ly, z + lz],
                    [x + lx, y - ly, z + lz],
                    [x + lx, y + ly, z + lz],
                    [x - lx, y + ly, z + lz],
                ]
            )
            obstacle_meshes.append((box_verts, vertex_offset))
            vertex_offset += 8
            total_added_obstacles += 1

        box_face_indices = [
            0, 1, 2, 3,  # bottom
            4, 5, 6, 7,  # top
            0, 1, 5, 4,  # front
            2, 3, 7, 6,  # back
            0, 3, 7, 4,  # left
            1, 2, 6, 5,  # right
        ]

        for verts, offset in obstacle_meshes:
            for vert in verts:
                existing_points.append(tuple(vert))
            for idx in box_face_indices:
                existing_face_indices.append(offset + idx)
            for _ in range(6):
                existing_face_counts.append(4)

        terrain_mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray(existing_points))
        terrain_mesh_prim.GetFaceVertexCountsAttr().Set(existing_face_counts)
        terrain_mesh_prim.GetFaceVertexIndicesAttr().Set(existing_face_indices)

        print(
            f"[ObstacleManager]: Added {total_added_obstacles} Static Obstacles (approximated as boxes) to Terrain Mesh"
        )
    else:
        logging.warning(
            f"No Terrain Mesh in {terrain_mesh_prim_path} , LiDAR will not detect static obstacles"
        )
    # ==================== (End of Modification) ====================


class DynamicObstacleManager:
    """
    管理动态障碍物的创建、状态和移动。
    """

    def __init__(self, cfg, map_range, device):
        self.cfg = cfg
        self.map_range = map_range
        self.device = device
        self.num_obstacles = self.cfg.env_dyn.num_obstacles

        if self.num_obstacles == 0:
            print("[ObstacleManager]: No Dynamic Obstacles")
            return

        print(f"[ObstacleManager]: Generating {self.num_obstacles} Dynamic Obstacles...")

        # 参数
        N_w = 4
        N_h = 2
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width / float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(
            self.num_obstacles / dyn_obs_category_num
        )
        # 修正总数以防舍入误差
        self.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num
        self.cfg.env_dyn.num_obstacles = self.num_obstacles

        # 状态信息
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros(
            (self.num_obstacles, 13), dtype=torch.float, device=self.device
        )
        self.dyn_obs_state[:, 3] = 1.0  # 四元数
        self.dyn_obs_goal = torch.zeros(
            (self.num_obstacles, 3), dtype=torch.float, device=self.device
        )
        self.dyn_obs_origin = torch.zeros(
            (self.num_obstacles, 3), dtype=torch.float, device=self.device
        )
        self.dyn_obs_vel = torch.zeros(
            (self.num_obstacles, 3), dtype=torch.float, device=self.device
        )
        self.dyn_obs_step_count = 0
        self.dyn_obs_size = torch.zeros(
            (self.num_obstacles, 3), dtype=torch.float, device=self.device
        )

        obs_dist = 2 * np.sqrt(
            self.map_range[0] * self.map_range[1] / self.num_obstacles
        )
        curr_obs_dist = obs_dist
        prev_pos_list = []
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num / N_h)

        for category_idx in range(cuboid_category_num + cylinder_category_num):
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                start_time = time.time()
                while True:
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if category_idx < cuboid_category_num:
                        oz = np.random.uniform(low=0.0, high=self.map_range[2])
                    else:
                        oz = self.max_obs_2d_height / 2.0
                    curr_pos = np.array([ox, oy])
                    valid = self._check_pos_validity(
                        prev_pos_list, curr_pos, curr_obs_dist
                    )
                    curr_time = time.time()
                    if curr_time - start_time > 0.1:
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if valid:
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                idx = origin_idx + category_idx * self.dyn_obs_num_of_each_category
                self.dyn_obs_origin[idx] = torch.tensor(
                    origin, dtype=torch.float, device=self.device
                )
                self.dyn_obs_state[idx, :3] = torch.tensor(
                    origin, dtype=torch.float, device=self.device
                )
                prim_utils.create_prim(f"/World/Origin{idx}", "Xform", translation=origin)

            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category

            if category_idx < cuboid_category_num:
                obs_width = width = float(category_idx + 1) * max_obs_width / float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(start_idx, end_idx)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=False
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = (
                    float(category_idx - cuboid_category_num + 1)
                    * max_obs_width
                    / float(N_w)
                    / 2.0
                )
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(start_idx, end_idx)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius=radius,
                        height=self.max_obs_2d_height,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=False
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)

            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[start_idx:end_idx] = torch.tensor(
                [obs_width, obs_width, obs_height],
                dtype=torch.float,
                device=self.device,
            )

    def _check_pos_validity(self, prev_pos_list, curr_pos, adjusted_obs_dist):
        """辅助函数，用于检查位置有效性。"""
        for prev_pos in prev_pos_list:
            if np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist:
                return False
        return True

    def update(self):
        """更新动态障碍物的位置。(等同于 move_dynamic_obstacle)"""
        if self.num_obstacles == 0:
            return

        # 步骤 1: 采样新目标点
        dyn_obs_goal_dist = (
            torch.sqrt(
                torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal) ** 2, dim=1)
            )
            if self.dyn_obs_step_count != 0
            else torch.zeros(self.dyn_obs_state.size(0), device=self.device)
        )
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5

        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2.0 * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2.0 * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.device)
        sample_z_local = -self.cfg.env_dyn.local_range[2] + 2.0 * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)

        self.dyn_obs_goal[dyn_obs_new_goal_mask] = (
            self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        )
        self.dyn_obs_goal[:, 0] = torch.clamp(
            self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0]
        )
        self.dyn_obs_goal[:, 1] = torch.clamp(
            self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1]
        )
        self.dyn_obs_goal[:, 2] = torch.clamp(
            self.dyn_obs_goal[:, 2], min=0.0, max=self.map_range[2]
        )
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0) / 2) :, 2] = (
            self.max_obs_2d_height / 2.0
        )

        # 步骤 2: 采样速度
        if self.dyn_obs_step_count % int(2.0 / self.cfg.sim.dt) == 0:
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (
                self.cfg.env_dyn.vel_range[1] - self.cfg.env_dyn.vel_range[0]
            ) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * (
                self.dyn_obs_goal - self.dyn_obs_state[:, :3]
            ) / torch.norm(
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True
            )

        # 步骤 3: 计算新位置
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt

        # 步骤 4: 更新模拟中的可视化
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category
            dynamic_obstacle.write_root_state_to_sim(
                self.dyn_obs_state[start_idx:end_idx]
            )
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1