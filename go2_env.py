import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class Go2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="/home/ansh/projekts/quadmove/unitree_go2/scene.xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # Action scaling from XML
        self.n_joints = self.model.nu
        self.action_low = self.model.actuator_ctrlrange[:, 0]
        self.action_high = self.model.actuator_ctrlrange[:, 1]

        # Observation space: joint pos/vel, base quat, lin vel, ang vel
        obs_dim = (
            self.model.nq - 7 +  # exclude root pos+quat
            self.model.nv - 6 +  # exclude root lin+ang vel
            4 +                  # base quat
            3 +                  # base lin vel
            3                    # base ang vel
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        
        # Episode tracking
        self.max_steps = 1000
        self.current_steps = 0

    def _get_obs(self):
        qpos = self.data.qpos[7:].ravel()
        qvel = self.data.qvel[6:].ravel()
        base_quat = self.data.qpos[3:7]
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        return np.concatenate([qpos, qvel, base_quat, base_linvel, base_angvel]).astype(np.float32)

    def step(self, action):
        # Rescale action from [-1, 1] to actuator range
        ctrl = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)
        self.current_steps += 1

        # --- Reward shaping ---
        forward_vel = self.data.qvel[0]
        forward_reward = forward_vel * 2.0  # Reward forward motion

        height = self.data.qpos[2]
        height_bonus = 1.0 - abs(height - 0.45) * 1.0  # Less punitive height reward

        # Reduced tilt penalty
        orientation_penalty = -np.sum(np.square(self.data.qpos[3:5])) * 0.5

        # Survival rewards
        alive_bonus = 0.5
        survival_bonus = 0.1  # Small reward for each step survived

        # Lower control cost
        ctrl_cost = 0.001 * np.square(action).sum()

        reward = (forward_reward + 
                 height_bonus + 
                 orientation_penalty + 
                 alive_bonus + 
                 survival_bonus - 
                 ctrl_cost)

        # Termination conditions (much more forgiving)
        height_termination = height < 0.15 or height > 0.8  # Wider bounds
        terminated = bool(height_termination)
        
        # Truncation after max steps
        truncated = self.current_steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_steps = 0

        # Add small random initial noise
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qvel[:] += np.random.uniform(-0.01, 0.01, size=self.model.nv)

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            # Use the older MuJoCo rendering approach
            try:
                # Try the modern viewer first
                import mujoco.viewer
                if not hasattr(self, 'viewer') or self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                else:
                    self.viewer.sync()
            except (AttributeError, ImportError):
                # Fallback to older rendering method
                if not hasattr(self, 'viewer') or self.viewer is None:
                    from mujoco_py import MjViewer
                    self.viewer = MjViewer()
                    self.viewer.set_model(self.model)
                    self.viewer.set_data(self.data)
                self.viewer.render()