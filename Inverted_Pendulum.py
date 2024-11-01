# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:52:46 2024

@author: YAKE

Sugita-lab Inverted Pendulum Env
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

class MFG_InvertedPendulum(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10,}
    
    def __init__(self, render_mode=None):
        self.model_path = 'InvertedPendulum.xml'
        self.m = mujoco.MjModel.from_xml_path(self.model_path)
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        
        self.init_qpos = np.copy(self.d.qpos)
        self.init_qvel = np.copy(self.d.qvel)
        
        self.dof_number = self.m.nv
        self.actuator_number = self.m.nu
        
        self.pos_upper_bounds = self.m.jnt_range[:, 1]
        self.pos_lower_bounds = self.m.jnt_range[:, 0]
        self.vel_upper_bounds = np.full(self.dof_number, np.inf)
        self.vel_lower_bounds = np.full(self.dof_number, -np.inf) 
        self.observation_space = spaces.Box(low=np.concatenate([self.pos_lower_bounds, self.vel_lower_bounds]), 
                                            high=np.concatenate([self.pos_upper_bounds, self.vel_upper_bounds]),
                                            shape=(self.dof_number * 2,), dtype=np.float64)
        
        self.ctrl_range = self.m.actuator_ctrlrange
        self.action_space = spaces.Box(low=self.ctrl_range[:, 0], high=self.ctrl_range[:, 1], shape=(self.actuator_number,), dtype=np.float64)
                
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.viewer = None
        
    def _get_obs(self):
        return np.concatenate([self.d.qpos, self.d.qvel]).ravel()

    def _compute_reward(self, action):
        upright_reward = 1 if -0.1 <= self.d.qpos[1] <= 0.1 else 0.0
        control_penalty = -0.01 * np.sum(np.square(action))
        velocity_penalty = -0.1 * np.abs(self.d.qvel[1])
        position_penalty = -0.5 * np.abs(self.d.qpos[0])
        return upright_reward + control_penalty + velocity_penalty + position_penalty
    
    def reset(self, seed=None, options=None):
        
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            
        mujoco.mj_resetData(self.m, self.d)
        
        noise_level_pos = 0.01
        noise_level_vel = 0.01
        initial_pos_noise = np.random.uniform(low=-noise_level_pos, high=noise_level_pos, size=self.dof_number)
        initial_vel_noise = np.random.uniform(low=-noise_level_vel, high=noise_level_vel, size=self.dof_number)
        
        self.d.qpos[:] = self.init_qpos + initial_pos_noise
        self.d.qvel[:] = self.init_qvel + initial_vel_noise
        
        if self.m.na == 0:
            self.d.act[:] = None
        mujoco.mj_forward(self.m, self.d)
        
        return self._get_obs(), {}
        
        
    def step(self, action):
        self.d.ctrl[:] = action
        mujoco.mj_step(self.m, self.d)
        obs = self._get_obs()
        
        out_of_bounds = not (self.observation_space.low <= obs).all() or not (obs <= self.observation_space.high).all()
        terminated = out_of_bounds or np.abs(obs[1]) > 0.25
        
        reward = self._compute_reward(action)
        
        if self.render_mode == "human":
            self.render()
        # print(obs, reward, terminated, False, {})
        return obs, reward, terminated, False, {}
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.m, self.d)
            
            time.sleep(self.dt*1)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()