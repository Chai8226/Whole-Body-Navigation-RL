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
    在场景中生成静态地形和带孔的垂直墙体。
    """
    percent_walls = float(getattr(cfg.env, "percent_hole", 0.5))

    print("[ObstacleManager]: Generating Obstacles (Walls with Holes)...")
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
                    num_obstacles=int(cfg.env.num_obstacles * (1-percent_walls)),
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

    # ==================== whole-body (修改为带孔的墙) ====================
    # 将带孔的垂直墙体添加到地形网格中

    num_walls = int(percent_walls * cfg.env.num_obstacles)
    wall_len_range = tuple(getattr(cfg.env, "wall_length_range", [3.0, 8.0]))
    wall_thk_range = tuple(getattr(cfg.env, "wall_thickness_range", [0.1, 0.3]))
    wall_hgt_range = tuple(getattr(cfg.env, "wall_height_range", [2.0, 4.0])) # 墙的最小高度为2.0，为孔留出空间

    # 新增：孔的参数
    hole_len_range = tuple(getattr(cfg.env, "wall_hole_length_range", [0.8, 2.0]))
    hole_hgt_range = tuple(getattr(cfg.env, "wall_hole_height_range", [0.8, 1.5]))
    hole_bottom_clearance = float(getattr(cfg.env, "wall_hole_bottom_clearance", 0.5)) # 孔距离地面的最小高度
    hole_side_clearance = float(getattr(cfg.env, "wall_hole_side_clearance", 0.5)) # 孔距离墙体边缘的最小距离

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

        # 定义一个带孔墙体(16个顶点)的局部面索引
        # 顶点顺序 (x, y, z 坐标):
        # 0-3:  外-底 (x-y-, x+y-, x+y+, x-y+)
        # 4-7:  外-顶 (x-y-, x+y-, x+y+, x-y+)
        # 8-11: 内-底 (x-y-, x+y-, x+y+, x-y+) (孔的底部)
        # 12-15: 内-顶 (x-y-, x+y-, x+y+, x-y+) (孔的顶部)
        
        local_face_indices_pattern = [
            # Front face (4 quads, 绕着孔)
            0, 1, 9, 8,      # 前-底
            12, 13, 5, 4,    # 前-顶
            0, 8, 12, 4,     # 前-左
            1, 5, 13, 9,     # 前-右
            # Back face (4 quads, 绕着孔)
            3, 2, 10, 11,    # 后-底
            15, 14, 6, 7,    # 后-顶
            3, 11, 15, 7,    # 后-左
            2, 6, 14, 10,    # 后-右
            # Outer Top/Bottom/Sides (4 quads)
            4, 5, 6, 7,      # 外-顶
            0, 1, 2, 3,      # 外-底
            0, 3, 7, 4,      # 外-左
            1, 2, 6, 5,      # 外-右
            # Inner Hole Faces (4 quads, 孔的“隧道”)
            12, 13, 14, 15,  # 内-顶
            8, 9, 10, 11,    # 内-底
            8, 11, 15, 12,   # 内-左
            9, 10, 14, 13,   # 内-右
        ]
        local_face_counts_pattern = [4] * 16 # 16 个四边形

        # 生成 N 个带孔的墙
        for i in range(num_walls):
            # 1. 采样墙体和孔的尺寸
            L = float(np.random.uniform(*wall_len_range))
            T = float(np.random.uniform(*wall_thk_range))
            H = float(np.random.uniform(*wall_hgt_range))
            
            L_h = float(np.random.uniform(*hole_len_range))
            H_h = float(np.random.uniform(*hole_hgt_range))
            
            # 2. 确保孔比墙小
            L_h = min(L_h, L - 2 * hole_side_clearance)
            H_h = min(H_h, H - hole_bottom_clearance - 0.1) # 0.1 for top clearance
            
            # 如果墙或孔太小，则跳过
            if L_h < 0.1 or H_h < 0.1:
                continue 

            # 3. 采样墙体中心位置 (x, y)
            x = float(np.random.uniform(-map_range[0] + L/2 + 1.5, map_range[0] - L/2 - 1.5))
            y = float(np.random.uniform(-map_range[1] + L/2 + 1.5, map_range[1] - L/2 - 1.5))
            
            # 4. 采样孔的中心位置 (相对于墙)
            # 孔在长度(L)上的偏移
            max_l_offset = (L - L_h) / 2 - hole_side_clearance
            l_offset = float(np.random.uniform(-max_l_offset, max_l_offset))
            
            # 孔在高度(H)上的Z坐标
            z_base = float(np.random.uniform(hole_bottom_clearance, H - H_h - 0.1)) # 0.1 for top clearance
            z_h_center = z_base + H_h / 2.0
            
            # 5. 决定朝向并计算16个顶点
            all_verts = np.zeros((16, 3))
            
            # 随机决定墙是 X-aligned 还是 Y-aligned
            if np.random.rand() > 0.5:
                # X-aligned 墙 (L 沿 X, T 沿 Y)
                lx_out, ly_out = L / 2, T / 2
                lxh_in, lyh_in = L_h / 2, T / 2 # 孔的厚度和墙一样
                
                x_h_c = x + l_offset # 孔的中心X
                y_h_c = y            # 孔的中心Y (同墙)
                
                x_out_min, x_out_max = x - lx_out, x + lx_out
                y_out_min, y_out_max = y - ly_out, y + ly_out
                
                x_in_min, x_in_max = x_h_c - lxh_in, x_h_c + lxh_in
                y_in_min, y_in_max = y_h_c - lyh_in, y_h_c + lyh_in

            else:
                # Y-aligned 墙 (L 沿 Y, T 沿 X)
                lx_out, ly_out = T / 2, L / 2
                lxh_in, lyh_in = T / 2, L_h / 2 # 孔的厚度和墙一样
                
                x_h_c = x            # 孔的中心X (同墙)
                y_h_c = y + l_offset # 孔的中心Y
                
                x_out_min, x_out_max = x - lx_out, x + lx_out
                y_out_min, y_out_max = y - ly_out, y + ly_out

                x_in_min, x_in_max = x_h_c - lxh_in, x_h_c + lxh_in
                y_in_min, y_in_max = y_h_c - lyh_in, y_h_c + lyh_in

            # Z 坐标
            z_out_min, z_out_max = 0.0, H
            z_in_min, z_in_max = z_base, z_base + H_h
            
            # 计算16个顶点
            # 0-3: 外-底
            all_verts[0] = [x_out_min, y_out_min, z_out_min]
            all_verts[1] = [x_out_max, y_out_min, z_out_min]
            all_verts[2] = [x_out_max, y_out_max, z_out_min]
            all_verts[3] = [x_out_min, y_out_max, z_out_min]
            # 4-7: 外-顶
            all_verts[4] = [x_out_min, y_out_min, z_out_max]
            all_verts[5] = [x_out_max, y_out_min, z_out_max]
            all_verts[6] = [x_out_max, y_out_max, z_out_max]
            all_verts[7] = [x_out_min, y_out_max, z_out_max]
            # 8-11: 内-底 (孔的底部)
            all_verts[8] = [x_in_min, y_in_min, z_in_min]
            all_verts[9] = [x_in_max, y_in_min, z_in_min]
            all_verts[10] = [x_in_max, y_in_max, z_in_min]
            all_verts[11] = [x_in_min, y_in_max, z_in_min]
            # 12-15: 内-顶 (孔的顶部)
            all_verts[12] = [x_in_min, y_in_min, z_in_max]
            all_verts[13] = [x_in_max, y_in_min, z_in_max]
            all_verts[14] = [x_in_max, y_in_max, z_in_max]
            all_verts[15] = [x_in_min, y_in_max, z_in_max]

            # 6. 将顶点和面添加到主网格
            for vert in all_verts:
                existing_points.append(tuple(vert))
            
            for idx in local_face_indices_pattern:
                existing_face_indices.append(vertex_offset + idx)
            
            existing_face_counts.extend(local_face_counts_pattern)
            
            vertex_offset += 16 # 每个墙体增加16个顶点

        terrain_mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray(existing_points))
        terrain_mesh_prim.GetFaceVertexCountsAttr().Set(existing_face_counts)
        terrain_mesh_prim.GetFaceVertexIndicesAttr().Set(existing_face_indices)

        print(
            f"[ObstacleManager]: 已添加 {num_walls} 个带孔墙体到地形网格。"
        )
    else:
        logging.warning(
            f"在 {terrain_mesh_prim_path} 未找到地形网格, LiDAR将无法检测到墙体！"
        )
    # ==================== whole-body (修改结束) ====================


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
            print("[ObstacleManager]: 不生成动态障碍物。")
            return

        print(f"[ObstacleManager]: 正在生成 {self.num_obstacles} 个动态障碍物...")

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