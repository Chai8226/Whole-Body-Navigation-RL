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
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
# Add body-frame rotation utility for shape-based dynamic obstacle handling
from omni_drones.utils.torch import quat_rotate_inverse

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)
    
    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
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

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        # Global curriculum step counter (counts env steps during training)
        self.curriculum_step = 0
        self._last_training_flag = bool(self.training)
        self._curriculum_step_checkpoint = 0

        # LiDAR Initialization (use precomputed vertical angles and beam count)
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=self.vertical_ray_angles_deg
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams_ext)
        
        # ==================== whole-body shape scan ====================
        # Compute robot shape scan: distance from center to surface along each ray
        self.shape_scan = self._compute_shape_scan(cfg.drone.marker.shape)
        print(f"Computed shape_scan with shape: {self.shape_scan.shape}")
        # ==================== whole-body shape scan ====================

        # start and target
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)

   
    # ==================== whole-body ====================
    def _compute_shape_scan(self, shape_name):
        """
        Compute distance from robot center to the outermost surface along each ray direction.
        This creates a shape scan with the same dimensions as lidar_scan.
        Returns the MAXIMUM distance along each ray to represent the robot's outer boundary.
        
        Args:
            shape_name: Name of the OBJ file (without .obj extension)
            
        Returns:
            torch.Tensor: Shape scan with dimensions (1, lidar_hbeams, lidar_vbeams_ext)
        """
        obj_folder = os.path.join(os.path.dirname(__file__), "..", "obj")
        obj_file = os.path.join(obj_folder, f"{shape_name}.obj")
        
        if not os.path.exists(obj_file):
            print(f"OBJ file not found: {obj_file}, using zero shape scan")
            return torch.zeros(1, self.lidar_hbeams, self.lidar_vbeams_ext, device=self.device)
        
        # Parse OBJ file to get vertices and faces
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
            print(f"Invalid OBJ file: {obj_file}, using zero shape scan")
            return torch.zeros(1, self.lidar_hbeams, self.lidar_vbeams_ext, device=self.device)
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Generate ray directions matching the LiDAR pattern
        # Horizontal angles: 0 to 360 degrees
        horizontal_angles = np.linspace(0, 2 * np.pi, self.lidar_hbeams, endpoint=False)
        # Vertical angles: use extended list including +/-90 deg
        vertical_angles = np.deg2rad(self.vertical_ray_angles_deg.cpu().numpy())
        
        shape_distances = np.zeros((self.lidar_hbeams, self.lidar_vbeams_ext))
        
        # For each ray direction, compute intersection with robot mesh
        for h_idx, h_angle in enumerate(horizontal_angles):
            for v_idx, v_angle in enumerate(vertical_angles):
                # Convert spherical to Cartesian coordinates
                # x: forward, y: left, z: up
                ray_dir = np.array([
                    np.cos(v_angle) * np.cos(h_angle),  # x
                    np.cos(v_angle) * np.sin(h_angle),  # y
                    np.sin(v_angle)                      # z
                ])
                
                # Find intersection with mesh
                max_distance = self._ray_mesh_intersection(
                    ray_origin=np.array([0.0, 0.0, 0.0]),
                    ray_direction=ray_dir,
                    vertices=vertices,
                    faces=faces
                )
                
                shape_distances[h_idx, v_idx] = max_distance
        
        # Convert to torch tensor
        shape_scan = torch.tensor(shape_distances, dtype=torch.float32, device=self.device)
        shape_scan = shape_scan.unsqueeze(0)  # Add batch dimension: (1, H, Vext)
        
        print(f"Shape scan statistics - min: {shape_scan.min():.4f}, max: {shape_scan.max():.4f}, mean: {shape_scan.mean():.4f}")
        
        # Save shape_scan data to file for visualization
        save_folder = os.path.join(os.path.dirname(__file__), "..", "shape_scan_data")
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"{shape_name}_shape_scan.npz")
        
        # Save both the shape_scan and metadata
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
        print(f"Shape scan saved to: {save_file}")
        
        return shape_scan
    
    def _ray_mesh_intersection(self, ray_origin, ray_direction, vertices, faces):
        """
        Compute the farthest intersection distance between a ray and a triangle mesh.
        Uses Möller-Trumbore ray-triangle intersection algorithm.
        This returns the maximum distance to represent the outermost surface of the robot.
        
        Args:
            ray_origin: Origin of the ray (3,)
            ray_direction: Direction of the ray (3,), should be normalized
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3), indices into vertices
            
        Returns:
            float: Distance to farthest intersection, or 0 if no intersection
        """
        epsilon = 1e-6
        max_distance = 0.0  # Changed from min_distance to max_distance
        
        # Normalize ray direction
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + epsilon)
        
        # Check intersection with each triangle
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Möller-Trumbore algorithm
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            # Ray is parallel to triangle
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
            
            # Compute intersection distance
            t = f * np.dot(edge2, q)
            
            # Valid intersection (in front of ray origin)
            # Take the MAXIMUM distance to get the outermost surface
            if t > epsilon:
                max_distance = max(max_distance, t)
        
        # Return 0 if no intersection found (ray doesn't hit the mesh)
        return max_distance
    # ==================== whole-body ====================

    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        # optional: attach a visualization-only marker at drone center (base_link)
        marker_cfg = getattr(self.cfg.drone, "marker", None)

        # ==================== whole-body visualization marker ====================
        # Now we load precise geometry from OBJ files instead of using simple shapes or points
        if marker_cfg is not None and bool(marker_cfg.enable):
                cfg.center_marker = True
                # shape: OBJ file name (without .obj extension)
                if hasattr(marker_cfg, "shape"):  
                    cfg.center_marker_shape = str(marker_cfg.shape)
                # color: RGB tuple
                if hasattr(marker_cfg, "color"):
                    cfg.center_marker_color = tuple(float(v) for v in marker_cfg.color)
                
                print(f"Enabling whole-body visualization with OBJ file: {cfg.center_marker_shape}.obj")
        else:
            cfg.center_marker = False
        # ====================================================================
        
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
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
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
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
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        # ==================== whole-body ====================
        # Add horizontal beams to the terrain mesh so LiDAR can detect them
        from pxr import UsdGeom
        from pxr import Vt
        
        num_beams = int(getattr(self.cfg.env, "num_static_beams", 12))
        beam_len_range = tuple(getattr(self.cfg.env, "beam_length_range", [2.0, 6.0]))
        beam_thk_range = tuple(getattr(self.cfg.env, "beam_thickness_range", [0.2, 0.4]))
        beam_z_range = tuple(getattr(self.cfg.env, "beam_height_range", [1.0, 3.0]))
        
        # Get the terrain mesh prim to append beam geometries
        stage = prim_utils.get_current_stage()
        terrain_mesh_prim_path = "/World/ground/terrain/mesh"
        terrain_mesh_prim = UsdGeom.Mesh.Get(stage, terrain_mesh_prim_path)
        
        if terrain_mesh_prim:
            # Get existing mesh data
            existing_points = list(terrain_mesh_prim.GetPointsAttr().Get())
            existing_face_counts = list(terrain_mesh_prim.GetFaceVertexCountsAttr().Get())
            existing_face_indices = list(terrain_mesh_prim.GetFaceVertexIndicesAttr().Get())
            vertex_offset = len(existing_points)
            
            # Generate beam meshes and append to terrain
            beam_meshes = []
            
            # X-direction beams
            for i in range(num_beams // 2 + num_beams % 2):
                L = float(np.random.uniform(*beam_len_range))
                T = float(np.random.uniform(*beam_thk_range))
                z = float(np.random.uniform(*beam_z_range))
                x = float(np.random.uniform(-self.map_range[0]+1.5, self.map_range[0]-1.5))
                y = float(np.random.uniform(-self.map_range[1]+1.5, self.map_range[1]-1.5))
                
                # Create box vertices
                lx, ly, lz = L/2, T/2, T/2
                box_verts = np.array([
                    [x-lx, y-ly, z-lz], [x+lx, y-ly, z-lz], [x+lx, y+ly, z-lz], [x-lx, y+ly, z-lz],
                    [x-lx, y-ly, z+lz], [x+lx, y-ly, z+lz], [x+lx, y+ly, z+lz], [x-lx, y+ly, z+lz]
                ])
                beam_meshes.append((box_verts, vertex_offset))
                vertex_offset += 8
            
            # Y-direction beams
            for i in range(num_beams // 2):
                L = float(np.random.uniform(*beam_len_range))
                T = float(np.random.uniform(*beam_thk_range))
                z = float(np.random.uniform(*beam_z_range))
                x = float(np.random.uniform(-self.map_range[0]+1.5, self.map_range[0]-1.5))
                y = float(np.random.uniform(-self.map_range[1]+1.5, self.map_range[1]-1.5))
                
                # Create box vertices
                lx, ly, lz = T/2, L/2, T/2
                box_verts = np.array([
                    [x-lx, y-ly, z-lz], [x+lx, y-ly, z-lz], [x+lx, y+ly, z-lz], [x-lx, y+ly, z-lz],
                    [x-lx, y-ly, z+lz], [x+lx, y-ly, z+lz], [x+lx, y+ly, z+lz], [x-lx, y+ly, z+lz]
                ])
                beam_meshes.append((box_verts, vertex_offset))
                vertex_offset += 8
            
            # Box face topology (same for all boxes)
            box_face_indices = [
                0, 1, 2, 3,  # bottom
                4, 5, 6, 7,  # top
                0, 1, 5, 4,  # front
                2, 3, 7, 6,  # back
                0, 3, 7, 4,  # left
                1, 2, 6, 5   # right
            ]
            
            # Append all beam vertices and faces to the terrain mesh
            for verts, offset in beam_meshes:
                for vert in verts:
                    existing_points.append(tuple(vert))
                for idx in box_face_indices:
                    existing_face_indices.append(offset + idx)
                for _ in range(6):  # 6 quad faces per box
                    existing_face_counts.append(4)
            
            # Update the terrain mesh with new geometry
            terrain_mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray(existing_points))
            terrain_mesh_prim.GetFaceVertexCountsAttr().Set(existing_face_counts)
            terrain_mesh_prim.GetFaceVertexIndicesAttr().Set(existing_face_indices)
            
            print(f"[Navigation Environment]: Added {num_beams} horizontal beams to terrain mesh for LiDAR detection")
        else:
            print(f"[WARNING]: Could not find terrain mesh at {terrain_mesh_prim_path}, beams will not be detected by LiDAR!")
        # ==================== whole-body ====================

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)

    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    # ==================== whole-body ====================
    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
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
        
        # Action Spec
        # Keep base env's action space as low-level motor commands; yaw control is introduced via VelController transform.
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # motor command space
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
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
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
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

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
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
        # Detect train/eval toggles and optionally freeze/restore curriculum counter during eval
        freeze_in_eval = bool(getattr(self.cfg.env, "curriculum_freeze_in_eval", True))
        if self.training != self._last_training_flag:
            if freeze_in_eval:
                if not self.training:
                    # entering eval: checkpoint current counter
                    self._curriculum_step_checkpoint = self.curriculum_step
                else:
                    # leaving eval back to training: restore counter (discard any eval increments)
                    self.curriculum_step = self._curriculum_step_checkpoint
            self._last_training_flag = bool(self.training)

        # advance global curriculum steps during training only
        if self.training:
            self.curriculum_step += 1
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    # ==================== whole-body ====================
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------

        # ==================== whole-body scan ====================
        # Step 1: Calculate distance from drone center to obstacles
        ray_distances = (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )
        
        # Debug: print raw lidar values at first step
        if self.progress_buf[0] == 0:
            print(f"[DEBUG] lidar_range: {self.lidar_range:.4f}m")
            print(f"[DEBUG] ray_distances - min: {ray_distances.min().item():.4f}, max: {ray_distances.max().item():.4f}, mean: {ray_distances.mean().item():.4f}")
            print(f"[DEBUG] shape_scan - min: {self.shape_scan.min().item():.4f}, max: {self.shape_scan.max().item():.4f}, mean: {self.shape_scan.mean().item():.4f}")
        
        clearance = (ray_distances - self.shape_scan).clamp(min=0.0)
        self.lidar_scan = clearance
        
        if self.progress_buf[0] == 0:
            print(f"[DEBUG] clearance (lidar_scan) - min: {self.lidar_scan.min().item():.4f}, max: {self.lidar_scan.max().item():.4f}, mean: {self.lidar_scan.mean().item():.4f}") 
        # ============================================================

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # Get size of dynamic obstacles (needed for whole-body clearance computation)
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the actual size

            # ==================== whole-body (shape-based clearance for dynamics) ====================
            # Compute robot radial extent along the direction to each obstacle by indexing shape_scan
            dir_vec = closest_dyn_obs_rpos / closest_dyn_obs_distance.clamp_min(1e-6)  # (N, K, 3), normalized
            # horizontal angle in [0, 2pi)
            phi = torch.atan2(dir_vec[..., 1], dir_vec[..., 0])
            two_pi = torch.tensor(2.0 * np.pi, device=self.device)
            phi = torch.remainder(phi + two_pi, two_pi)
            H = self.lidar_hbeams
            V = self.lidar_vbeams_ext
            # vertical angle in radians
            v_angle = torch.atan2(dir_vec[..., 2], torch.norm(dir_vec[..., :2], dim=-1))
            deg = v_angle * (180.0 / np.pi)
            vfov_min = -90.0
            vfov_max = 90.0
            # map angles to discrete beam indices
            h_idx = torch.round(phi / two_pi * (H)).long() % H
            v_idx = torch.round((deg - vfov_min) / max(vfov_max - vfov_min, 1e-6) * (V - 1)).clamp(0, V - 1).long()
            # gather per-direction robot radial distance from shape_scan
            shape_map = self.shape_scan[0].reshape(H * V)  # (H*V)
            lin_idx = (h_idx * V + v_idx).view(-1)  # flatten to (N*K)
            shape_r = shape_map[lin_idx].view(h_idx.shape)  # (N, K)
            shape_r = shape_r.unsqueeze(-1)  # (N, K, 1)
            # Clearance to obstacle surfaces (3D, with 2D-only already encoded by z=0 for last half)
            width_half = (closest_dyn_obs_size[..., 0].unsqueeze(-1)) * 0.5  # (N, K, 1)
            clearance_3d = (closest_dyn_obs_distance - shape_r - width_half).clamp_min(0.0)
            # 2D clearance in goal frame (use same radial extent approximation)
            dist2d_center = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            clearance_2d = (dist2d_center - shape_r - width_half).clamp_min(0.0)
            # Override distance features to be surface-clearance aware
            closest_dyn_obs_distance_2d = clearance_2d
            # keep z term as-is (optional: could subtract vertical extent if available)
            # Provide clearance tensor for reward below via a reserved name
            closest_dyn_obs_clearance_reward = clearance_3d.squeeze(-1)
            # Mask out-of-range obstacles
            closest_dyn_obs_clearance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            # ==================== whole-body ====================

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category (already defined above for whole-body clearance)
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # remove early dynamic-collision heuristics; collision is computed later from clearance
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states,
            # ==================== whole-body shape scan ====================
            "shape_scan": self.shape_scan.expand(self.num_envs, -1, -1, -1)  # Expand to batch size
            # ===============================================================
        }


        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        # ==================== whole-body ====================
        # Using exponential decay form: λ(1 - e^(-k*d²))
        # Advantages: bounded at d→0, smooth gradients, fast saturation at large d
        safety_lambda = getattr(self.cfg.env, "safety_reward_lambda", 1.0)
        safety_k = getattr(self.cfg.env, "safety_reward_k", 0.5)
        
        # Compute exponential safety reward for each clearance distance
        clearance_squared = self.lidar_scan ** 2
        reward_safety_static = safety_lambda * (1.0 - torch.exp(-safety_k * clearance_squared))
        reward_safety_static = reward_safety_static.mean(dim=(2, 3))  # Average over all rays
        # ==================== whole-body ====================
        
        # ==================== whole-body ====================
        # b. safety reward for dynamic obstacles
        if (self.cfg.env_dyn.num_obstacles != 0):
            # Use clearance-based distance with exponential decay: λ(1 - e^(-k*d²))
            clearance_dyn_squared = closest_dyn_obs_clearance_reward ** 2
            reward_safety_dynamic = safety_lambda * (1.0 - torch.exp(-safety_k * clearance_dyn_squared))
            reward_safety_dynamic = reward_safety_dynamic.mean(dim=-1, keepdim=True)
        # ==================== whole-body ====================

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)

        # ==================== whole-body ====================
        # e. height penalty reward for flying unnecessarily high or low
        # Using Huber loss for better gradient stability and training dynamics
        height_deadzone = getattr(self.cfg.env, "height_penalty_deadzone", 0.2)
        huber_delta = getattr(self.cfg.env, "height_penalty_huber_delta", 0.5)
        
        # Compute height violations
        drone_height = self.drone.pos[..., 2]
        upper_violation = torch.clamp(drone_height - (self.height_range[..., 1] + height_deadzone), min=0.0)
        lower_violation = torch.clamp((self.height_range[..., 0] - height_deadzone) - drone_height, min=0.0)
        
        # Huber loss: smooth L1 that transitions from quadratic to linear
        def huber_loss(x, delta):
            abs_x = torch.abs(x)
            quadratic = torch.where(abs_x <= delta, 0.5 * x**2, torch.zeros_like(x))
            linear = torch.where(abs_x > delta, delta * (abs_x - 0.5 * delta), torch.zeros_like(x))
            return quadratic + linear
        
        penalty_height = huber_loss(upper_violation, huber_delta) + huber_loss(lower_violation, huber_delta)
        # ==================== whole-body ====================


        # f. Collision condition with its penalty
        # ==================== whole-body ====================
        # Static collision already uses surface distance via lidar_scan
        lidar_min = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
        
        # Relaxed collision threshold: use 0.1m, and skip first 3 steps
        collision_threshold = 0.1
        static_collision = (lidar_min < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)
        # Dynamic collision: surface clearance in 3D less than margin
        if (self.cfg.env_dyn.num_obstacles != 0):
            dynamic_collision = (closest_dyn_obs_clearance_reward.min(dim=1, keepdim=True).values < collision_threshold) & (self.progress_buf.unsqueeze(1) > 5)
        else:
            dynamic_collision = torch.zeros_like(static_collision)
        # ==================== whole-body ====================
        
        collision = static_collision | dynamic_collision
        
        # ==================== whole-body shape scan ====================
        # Final reward calculation with curriculum weighting
        # Get reward weights and curriculum settings from config (with defaults)
        height_penalty_weight = getattr(self.cfg.env, "height_penalty_weight", 8.0)

        # Curriculum configuration
        # Option A (preferred if set): piecewise by two step markers r1 and r2
        r1 = getattr(self.cfg.env, "curriculum_r1_steps", None)
        r2 = getattr(self.cfg.env, "curriculum_r2_steps", None)

        # Option B (fallback): continuous schedule with total_steps and step_divisor
        total_steps = float(getattr(self.cfg.env, "curriculum_total_steps", 1_000_000))
        step_divisor = float(getattr(self.cfg.env, "curriculum_step_divisor", 1.0))

        # Progress p in [0,1]: 0 at start (more velocity), 1 at end (more safety)
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
                # fallback to continuous schedule
                effective_steps = self.curriculum_step / max(step_divisor, 1e-9)
                p = float(min(1.0, max(0.0, effective_steps / max(total_steps, 1e-9))))
        else:
            p = 1.0  # evaluation uses final weights (safety-focused)

        # Linear schedules for weights
        vel_w_start = float(getattr(self.cfg.env, "vel_weight_start", 1.5))
        vel_w_end = float(getattr(self.cfg.env, "vel_weight_end", 1.0))
        safety_w_start = float(getattr(self.cfg.env, "safety_weight_start", 0.8))
        safety_w_end = float(getattr(self.cfg.env, "safety_weight_end", 1.5))
        safety_dyn_w_start = float(getattr(self.cfg.env, "safety_dyn_weight_start", safety_w_start))
        safety_dyn_w_end = float(getattr(self.cfg.env, "safety_dyn_weight_end", safety_w_end))

        w_vel = vel_w_start + (vel_w_end - vel_w_start) * p
        w_safety_static = safety_w_start + (safety_w_end - safety_w_start) * p
        w_safety_dynamic = safety_dyn_w_start + (safety_dyn_w_end - safety_dyn_w_start) * p

        base_bias = 1.0

        # Optional curriculum logging: every N steps in training, once at start in eval
        log_interval = int(getattr(self.cfg.env, "curriculum_log_interval", 1000))
        should_log_train = self.training and (log_interval > 0) and (self.curriculum_step % log_interval == 0)
        should_log_eval = (not self.training) and (self.progress_buf[0].item() == 0)
        if should_log_train or should_log_eval:
            # Use effective step for display to avoid confusion during eval
            effective_step_display = (
                self.curriculum_step if self.training else (self._curriculum_step_checkpoint if log_interval >= 0 else self.curriculum_step)
            )
            if (self.cfg.env_dyn.num_obstacles != 0):
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

        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = (
                reward_vel * w_vel
                + base_bias
                + reward_safety_static * w_safety_static
                + reward_safety_dynamic * w_safety_dynamic
                - penalty_smooth * 0.1
                - penalty_height * height_penalty_weight
            )
        else:
            self.reward = (
                reward_vel * w_vel
                + base_bias
                + reward_safety_static * w_safety_static
                - penalty_smooth * 0.1
                - penalty_height * height_penalty_weight
            )
        # ==================== whole-body shape scan ====================
        

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next iteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # -----------------Training Stats-----------------
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
