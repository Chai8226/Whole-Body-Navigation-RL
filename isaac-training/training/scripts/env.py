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
# 删除了 Terrain 和 RigidObject 的导入
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
# 删除了 RigidObject 的导入
import time
# 为基于形状的动态障碍物处理添加机身坐标系旋转工具
from omni_drones.utils.torch import quat_rotate_inverse
import importlib


# from obs_hole import spawn_static_obstacles, DynamicObstacleManager



class NavigationEnv(IsaacEnv):

    # 在一个步骤中:
    # 1. _pre_sim_step (应用动作) -> 步进 isaac sim
    # 2. _post_sim_step (更新 lidar)
    # 3. 增加 progress_buf
    # 4. _compute_state_and_obs (获取观测和状态, 更新统计数据)
    # 5. _compute_reward_and_done (更新奖励并计算回报)
    
    def __init__(self, cfg):
        print("[Navigation Environment]: Init Environment...")
        # LiDAR 参数:
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


# === 在这里添加动态导入代码 (正确的位置) ===
        
        # 1. 从传入的 'cfg' 参数中获取 map_name
        #    我们在这里使用局部变量 'cfg'，因为它在 __init__ 的开头立即可用。
        map_name = cfg.env.map_name 
        obstacle_module_name = f"obs_{map_name}"
        print(f"[Navigation Environment]: Loading Map: {obstacle_module_name}.py")
        
        try:
            # 2. 动态导入指定的模块
            obs_module = importlib.import_module(obstacle_module_name)
        except ImportError as e:
            # 导入默认模块作为后备 (假设你有一个 'obs_vanilla.py' 作为备用)
            logging.error(f"Error: Can not load Map '{obstacle_module_name}'. Back to 'obs_vanilla'. Error: {e}")
            import obs_vanilla as obs_module 
        
        # 3. 将导入的函数和类存储在 'self' 上
        #    这样 _design_scene 才能在 super().__init__ 内部找到它们
        self.spawn_static_obstacles = obs_module.spawn_static_obstacles
        self.DynamicObstacleManager = obs_module.DynamicObstacleManager
        # --- 动态导入结束 ---
        
        super().__init__(cfg, cfg.headless)

        # 无人机初始化
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        # 全局课程步数计数器 (计算训练期间的环境步数)
        self.curriculum_step = 0
        self._last_training_flag = bool(self.training)
        self._curriculum_step_checkpoint = 0

        # LiDAR 初始化 (使用预先计算的垂直角度和光束数)
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
        # 计算机器人形状扫描：从中心到每个射线方向上最外层表面的距离
        self.shape_scan = self._compute_shape_scan(cfg.drone.marker.shape)
        print(f"shape_scan: {self.shape_scan.shape}")
        # ==================== whole-body shape scan ====================

        # 起点和目标
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # 坐标变更：添加目标方向变量
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
        
        # 解析 OBJ 文件以获取顶点和面
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
        
        # 生成匹配 LiDAR 模式的射线方向
        # 水平角度: 0 到 360 度
        horizontal_angles = np.linspace(0, 2 * np.pi, self.lidar_hbeams, endpoint=False)
        # 垂直角度: 使用包括 +/-90 度的扩展列表
        vertical_angles = np.deg2rad(self.vertical_ray_angles_deg.cpu().numpy())
        
        shape_distances = np.zeros((self.lidar_hbeams, self.lidar_vbeams_ext))
        
        # 对于每个射线方向，计算与机器人网格的交点
        for h_idx, h_angle in enumerate(horizontal_angles):
            for v_idx, v_angle in enumerate(vertical_angles):
                # 将球坐标转换为笛卡尔坐标
                # x: 前, y: 左, z: 上
                ray_dir = np.array([
                    np.cos(v_angle) * np.cos(h_angle),  # x
                    np.cos(v_angle) * np.sin(h_angle),  # y
                    np.sin(v_angle)                      # z
                ])
                
                # 查找与网格的交点
                max_distance = self._ray_mesh_intersection(
                    ray_origin=np.array([0.0, 0.0, 0.0]),
                    ray_direction=ray_dir,
                    vertices=vertices,
                    faces=faces
                )
                
                shape_distances[h_idx, v_idx] = max_distance
        
        # 转换为 torch 张量
        shape_scan = torch.tensor(shape_distances, dtype=torch.float32, device=self.device)
        shape_scan = shape_scan.unsqueeze(0)  # 添加批次维度: (1, H, Vext)
        
        print(f"Shape Scan Statistics - Min: {shape_scan.min():.4f}, Max: {shape_scan.max():.4f}, Avg: {shape_scan.mean():.4f}")
        
        # 将 shape_scan 数据保存到文件以便可视化
        save_folder = os.path.join(os.path.dirname(__file__), "..", "shape_scan_data")
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"{shape_name}_shape_scan.npz")
        
        # 保存 shape_scan 和元数据
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
        """
        计算射线与三角网格之间的最远交点距离。
        使用 Möller-Trumbore 射线-三角形相交算法。
        这返回最大距离以表示机器人的最外层表面。
        
        Args:
            ray_origin: 射线的起点 (3,)
            ray_direction: 射线的方向 (3,), 应归一化
            vertices: 网格顶点 (N, 3)
            faces: 网格面 (M, 3), 顶点的索引
            
        Returns:
            float: 到最远交点的距离，如果没有交点则为 0
        """
        epsilon = 1e-6
        max_distance = 0.0  # 从 min_distance 更改为 max_distance
        
        # 归一化射线方向
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + epsilon)
        
        # 检查与每个三角形的交点
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Möller-Trumbore 算法
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            # 射线与三角形平行
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
            
            # 计算交点距离
            t = f * np.dot(edge2, q)
            
            # 有效交点 (在射线起点前方)
            # 取最大距离以获得最外层表面
            if t > epsilon:
                max_distance = max(max_distance, t)
        
        # 如果未找到交点 (射线未击中网格)，则返回 0
        return max_distance
    # ==================== whole-body ====================

    def _design_scene(self):
        # 在 prim /World/envs/envs_0 中初始化无人机
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # 无人机模型类
        cfg = drone_model.cfg_cls(force_sensor=False)
        # 可选：在无人机中心 (base_link) 附加一个仅用于可视化的标记
        marker_cfg = getattr(self.cfg.drone, "marker", None)

        # ==================== whole-body visualization marker ====================
        # 现在我们从 OBJ 文件加载精确的几何形状，而不是使用简单的形状或点
        if marker_cfg is not None and bool(marker_cfg.enable):
                cfg.center_marker = True
                # shape: OBJ 文件名 (不带 .obj 扩展名)
                if hasattr(marker_cfg, "shape"):  
                    cfg.center_marker_shape = str(marker_cfg.shape)
                # color: RGB 元组
                if hasattr(marker_cfg, "color"):
                    cfg.center_marker_color = tuple(float(v) for v in marker_cfg.color)
                
                print(f"Enable Whole Body Visualization, Using OBJ File: {cfg.center_marker_shape}.obj")
        else:
            cfg.center_marker = False
        # ====================================================================
        
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # 灯光
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
        
        # 地平面
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        # --- 开始重构 ---
        # 生成静态障碍物 (地形和横梁)
        self.spawn_static_obstacles(self.cfg, self.num_envs, self.map_range)

        # 生成动态障碍物
        self.dyn_obs_manager = self.DynamicObstacleManager(self.cfg, self.map_range, self.device)
        # --- 结束重构 ---


    # 函数 move_dynamic_obstacle(self) 已被移除
    # 其逻辑现在位于 DynamicObstacleManager.update()


    # ==================== whole-body ====================
    def _set_specs(self):
        observation_dim = 7  # 从 8 更改: 3(rpos_clipped_b) + 1(distance) + 3(vel_b)
        num_dim_each_dyn_obs_state = 9  # 从 10 更改: 3(rpos_gn) + 1(distance) + 3(vel_g) + 1(width) + 1(height)

        # 观测空间
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams_ext), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                    # ==================== whole-body ====================
                    "shape_scan": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams_ext), device=self.device),
                    # ====================================================
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # 动作空间
        # 保持基础环境的动作空间为低级电机命令；偏航控制通过 VelController 变换引入。
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # 电机命令空间
            })
        }).expand(self.num_envs).to(self.device)
        
        # 奖励空间
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # 完成信号空间
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
            # wandb visualization for rewards
            "ep_reward_vel": UnboundedContinuousTensorSpec(1),
            "ep_reward_safety_static": UnboundedContinuousTensorSpec(1),
            "ep_reward_safety_dynamic": UnboundedContinuousTensorSpec(1),
            "ep_reward_height": UnboundedContinuousTensorSpec(1),
            "ep_penalty_smooth": UnboundedContinuousTensorSpec(1),
            "ep_bias": UnboundedContinuousTensorSpec(1),
            "ep_reward_reach_goal": UnboundedContinuousTensorSpec(1),
            "ep_penalty_collision": UnboundedContinuousTensorSpec(1),
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
            # 决定在哪一边
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # 生成随机位置
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights
            target_pos = target_pos * selected_masks + selected_shifts
            
            # 应用目标位置
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

            # 生成随机位置
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights
            pos = pos * selected_masks + selected_shifts
            
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # 坐标变更：重置后，无人机的目标方向应改变
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # 坐标变更：重置后，无人机的朝向应面向当前目标
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
        # 检测训练/评估切换，并在评估期间选择性地冻结/恢复课程计数器
        freeze_in_eval = bool(getattr(self.cfg.env, "curriculum_freeze_in_eval", True))
        if self.training != self._last_training_flag:
            if freeze_in_eval:
                if not self.training:
                    # 进入评估：检查点当前计数器
                    self._curriculum_step_checkpoint = self.curriculum_step
                else:
                    # 离开评估返回训练：恢复计数器 (丢弃任何评估增量)
                    self.curriculum_step = self._curriculum_step_checkpoint
            self._last_training_flag = bool(self.training)

        # 仅在训练期间推进全局课程步数
        if self.training:
            self.curriculum_step += 1
        
        # --- 开始重构 ---
        # 使用管理器更新动态障碍物
        self.dyn_obs_manager.update()
        # --- 结束重构 ---

        self.lidar.update(self.dt)
    
    # ==================== whole-body ====================
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info 用于控制器

        # -----------网络输入 I: LiDAR 范围数据--------------

        # 步骤 1: 计算从无人机中心到障碍物的距离
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

        # LiDAR 的可选渲染
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------网络输入 II: 无人机内部状态---------
        # 获取无人机当前的姿态四元数
        quat_w = self.root_state[..., 3:7]

        # a. 水平和垂直平面上的距离信息
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # 到目标的距离
        
        # b. 指向目标的单位方向向量
        target_dir_3d = self.target_dir.clone()

        rpos_clipped = rpos / distance.clamp(1e-6) # 单位向量：指向目标的方向
        rpos_clipped_b = quat_rotate_inverse(quat_w, rpos_clipped)
        
        # c. 无人机机身坐标系中的速度
        vel_w = self.root_state[..., 7:10] # 世界坐标系速度
        vel_b = quat_rotate_inverse(quat_w, vel_w)   # 速度的坐标变换

        # 最终的无人机内部状态
        drone_state = torch.cat([rpos_clipped_b, distance, vel_b], dim=-1).squeeze(1)

        # --- 开始重构 ---
        # 使用障碍物管理器获取信息
        if (self.dyn_obs_manager.num_obstacles > 0):
            # ---------网络输入 III: 动态障碍物状态--------
            # ------------------------------------------------------------
            # a. 目标坐标系中最近的 N 个障碍物的相对位置
            # 为每个无人机找到 N 个最近且在范围内的障碍物
            dyn_obs_pos_expanded = self.dyn_obs_manager.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_distance = torch.norm(dyn_obs_rpos_expanded, dim=2)  # 形状: (num_envs, num_dyn_obs)
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # 挑选最近的 N 个障碍物索引
            dyn_obs_range_mask = dyn_obs_distance.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # 无人机机身坐标系中障碍物的相对距离
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_3d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # 排除范围外的障碍物
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # 获取动态障碍物的尺寸 (用于全身间隙计算)
            closest_dyn_obs_size = self.dyn_obs_manager.dyn_obs_size[closest_dyn_obs_idx] # 实际尺寸

            # ==================== whole-body ====================
            # 通过索引 shape_scan 计算机器人沿每个障碍物方向的径向范围
            dir_vec = closest_dyn_obs_rpos / closest_dyn_obs_distance.clamp_min(1e-6)  # (N, K, 3), 归一化
            phi = torch.atan2(dir_vec[..., 1], dir_vec[..., 0])
            two_pi = torch.tensor(2.0 * np.pi, device=self.device)
            phi = torch.remainder(phi + two_pi, two_pi)
            # phi = torch.fmod(phi + two_pi, two_pi)
            H = self.lidar_hbeams
            V = self.lidar_vbeams_ext
            # 垂直角度 (弧度)
            v_angle = torch.atan2(dir_vec[..., 2], torch.norm(dir_vec[..., :2], dim=-1))
            deg = v_angle * (180.0 / np.pi)
            vfov_min = -90.0
            vfov_max = 90.0
            # 将角度映射到离散的光束索引
            h_idx = torch.round(phi / two_pi * (H)).long() % H
            v_idx = torch.round((deg - vfov_min) / max(vfov_max - vfov_min, 1e-6) * (V - 1)).clamp(0, V - 1).long()
            # 从 shape_scan 中收集每个方向的机器人径向距离
            shape_map = self.shape_scan[0].reshape(H * V)  # (H*V)
            lin_idx = (h_idx * V + v_idx).view(-1)  # 展平为 (N*K)
            shape_r = shape_map[lin_idx].view(h_idx.shape)  # (N, K)
            shape_r = shape_r.unsqueeze(-1)  # (N, K, 1)
            # 到障碍物表面的间隙 (3D, 2D-only 已通过 z=0 编码)
            width_half = (closest_dyn_obs_size[..., 0].unsqueeze(-1)) * 0.5  # (N, K, 1)
            clearance_3d = (closest_dyn_obs_distance - shape_r - width_half).clamp_min(0.0)
            
            # 通过保留名称为下面的奖励提供间隙张量
            closest_dyn_obs_clearance_reward = clearance_3d.squeeze(-1)
            # 屏蔽范围外的障碍物
            closest_dyn_obs_clearance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            # ==================== whole-body ====================

            # b. 动态障碍物在目标坐标系中的速度
            closest_dyn_obs_vel = self.dyn_obs_manager.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_3d) 

            # c. 类别中的动态障碍物尺寸 (已在上面为全身间隙定义)
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # 转换为类别: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # 连接所有动态障碍物信息
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)      
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 9, device=self.cfg.device)
        # --- 结束重构 ---
            
        # -----------------网络输入最终----------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_3d,
            "dynamic_obstacle": dyn_obs_states,
            # ==================== whole-body shape scan ====================
            "shape_scan": self.shape_scan.expand(self.num_envs, -1, -1, -1)  # 扩展到批次大小
            # ===============================================================
        }


        # -----------------奖励计算-----------------
        # a. 静态障碍物的安全奖励

        # ==================== whole-body ====================
        safety_lambda = getattr(self.cfg.env, "safety_reward_lambda", 1.0)
        safety_k = getattr(self.cfg.env, "safety_reward_k", 0.5)
        clearance_squared = self.lidar_scan ** 2
        reward_safety_static = safety_lambda * (1.0 - torch.exp(-safety_k * clearance_squared))
        reward_safety_static = reward_safety_static.mean(dim=(2, 3))  # 对所有射线取平均
        # wandb visualization for rewards
        reward_safety_dynamic = torch.zeros_like(reward_safety_static)
        # b. 动态障碍物的安全奖励
        # --- 开始重构 ---
        if (self.dyn_obs_manager.num_obstacles > 0):
        # --- 结束重构 ---
            clearance_dyn_squared = closest_dyn_obs_clearance_reward ** 2
            reward_safety_dynamic = safety_lambda * (1.0 - torch.exp(-safety_k * clearance_dyn_squared))
            reward_safety_dynamic = reward_safety_dynamic.mean(dim=-1, keepdim=True)
        # ==================== whole-body ====================

        # c. 朝向目标方向的速度奖励
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)  # (num_envs, 1)
        
        # d. 动作平滑度的平滑奖励
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)  # (num_envs, 1)

        # ==================== whole-body ====================
        # e. 统一高度奖励：结合了对最佳高度的偏好和对越界的惩罚
        # 最佳高度被动态设置为目标的 z 坐标。
        optimal_height = self.target_pos[:, 0, 2]  # (num_envs,)
        height_sigma = getattr(self.cfg.env, "height_reward_sigma", 0.5)
        
        drone_height = self.drone.pos[..., 2]  # (num_envs,) or (num_envs, 1)
        if drone_height.dim() > 1:
            drone_height = drone_height.squeeze(-1)  # 确保 (num_envs,)
        height_diff = drone_height - optimal_height  # (num_envs,)
        reward_height_pref = torch.exp(-0.5 * (height_diff / height_sigma) ** 2)  # (num_envs,)
        
        min_h = self.height_range[:, 0, 0]  # (num_envs,)
        max_h = self.height_range[:, 0, 1]  # (num_envs,)
        out_of_bounds_lower = torch.clamp(min_h - drone_height, min=0.0)  # (num_envs,)
        out_of_bounds_upper = torch.clamp(drone_height - max_h, min=0.0)  # (num_envs,)
        penalty_out_of_bounds = out_of_bounds_lower**2 + out_of_bounds_upper**2  # (num_envs,)
        
        # 将偏好和惩罚组合成一个单一的高度奖励项
        height_reward_weight = getattr(self.cfg.env, "height_reward_weight", 1.0)
        height_penalty_weight = getattr(self.cfg.env, "height_penalty_weight", 4.0)
        # reward_height = (reward_height_pref * height_reward_weight - penalty_out_of_bounds * height_penalty_weight).unsqueeze(-1)  # (num_envs, 1)
        reward_height = (- penalty_out_of_bounds * height_penalty_weight).unsqueeze(-1)  # (num_envs, 1)

        # ==================== whole-body ====================


        # f. 碰撞条件及其惩罚
        # ==================== whole-body ====================
        # 静态碰撞已通过 lidar_scan 使用表面距离
        lidar_min = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
        collision_threshold = 0.1
        static_collision = (lidar_min < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)

        # --- 开始重构 ---
        if (self.dyn_obs_manager.num_obstacles > 0):
        # --- 结束重构 ---
            dynamic_collision = (closest_dyn_obs_clearance_reward.min(dim=1, keepdim=True).values < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)
        else:
            dynamic_collision = torch.zeros_like(static_collision)
        # ==================== whole-body ====================
        
        collision = static_collision | dynamic_collision
        penalty_collision = (collision.float() * -10.0)
        
        # ==================== whole-body ====================
        # 课程配置
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
                # 回退到连续时间表
                effective_steps = self.curriculum_step / max(step_divisor, 1e-9)
                p = float(min(1.0, max(0.0, effective_steps / max(total_steps, 1e-9))))
        else:
            p = 1.0  # 评估使用最终权重 (注重安全)

        # 权重的线性调度
        vel_w_start = float(getattr(self.cfg.env, "vel_weight_start", 1.5))
        vel_w_end = float(getattr(self.cfg.env, "vel_weight_end", 1.0))
        safety_w_start = float(getattr(self.cfg.env, "safety_weight_start", 0.8))
        safety_w_end = float(getattr(self.cfg.env, "safety_weight_end", 1.5))
        safety_dyn_w_start = float(getattr(self.cfg.env, "safety_dyn_weight_start", safety_w_start))
        safety_dyn_w_end = float(getattr(self.cfg.env, "safety_dyn_weight_end", safety_w_end))

        w_vel = vel_w_start + (vel_w_end - vel_w_start) * p
        w_safety_static = safety_w_start + (safety_w_end - safety_w_start) * p
        w_safety_dynamic = safety_dyn_w_start + (safety_dyn_w_end - safety_dyn_w_start) * p

        base_bias = 0.5

        # 可选的课程日志记录：训练中每 N 步记录一次，评估开始时记录一次
        log_interval = int(getattr(self.cfg.env, "curriculum_log_interval", 1000))
        should_log_train = self.training and (log_interval > 0) and (self.curriculum_step % log_interval == 0)
        should_log_eval = (not self.training) and (self.progress_buf[0].item() == 0)
        if should_log_train or should_log_eval:
            # 使用有效步数进行显示，以避免评估期间产生混淆
            effective_step_display = (
                self.curriculum_step if self.training else (self._curriculum_step_checkpoint if log_interval >= 0 else self.curriculum_step)
            )
            # --- 开始重构 ---
            if (self.dyn_obs_manager.num_obstacles > 0):
            # --- 结束重构 ---
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
         # g. 到达目标的奖励
        reach_goal = (distance.squeeze(-1) < 0.5)
        reward_reach_goal = reach_goal.float() * 10.0

        r_vel = reward_vel * w_vel
        r_bias = torch.full_like(r_vel, base_bias)
        r_safety_static = reward_safety_static * w_safety_static
        r_penalty_smooth = -penalty_smooth * 0.1

        # --- 开始重构 ---
        if (self.dyn_obs_manager.num_obstacles > 0):
            r_safety_dynamic = reward_safety_dynamic * w_safety_dynamic
        # --- 结束重构 ---
            self.reward = (
                r_vel
                + base_bias
                + r_safety_static
                + r_safety_dynamic
                + r_penalty_smooth
                # - penalty_out_of_bounds * height_penalty_weight
                + reward_height 
                # + reward_reach_goal
                # + penalty_collision
            )
        else:
            r_safety_dynamic = torch.zeros_like(r_vel)
            self.reward = (
                r_vel
                + base_bias
                + r_safety_static
                + r_penalty_smooth
                # - penalty_out_of_bounds * height_penalty_weight
                + reward_height 
                # + reward_reach_goal
                # + penalty_collision
            )
        # ==================== whole-body ====================
        

        # 终止条件
        # reach_goal = (distance.squeeze(-1) < 0.5)

        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf 用于跟踪步数

        # 更新先前的速度，用于下一次迭代的平滑度计算
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------训练统计-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        # wandb visualization for rewards
        self.stats["ep_reward_vel"] += r_vel
        self.stats["ep_reward_safety_static"] += r_safety_static
        self.stats["ep_reward_safety_dynamic"] += r_safety_dynamic
        self.stats["ep_reward_height"] += reward_height
        self.stats["ep_penalty_smooth"] += r_penalty_smooth
        self.stats["ep_bias"] += r_bias
        self.stats["ep_reward_reach_goal"] += reward_reach_goal
        self.stats["ep_penalty_collision"] += penalty_collision

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