# src/modified_cartpole.py
import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium import logger
import numpy as np

class ModifiedCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super(ModifiedCartPoleEnv, self).__init__(render_mode=render_mode, sutton_barto_reward=False)
        # Extend the action space from 2 to 3
        self.action_space = gym.spaces.Discrete(3)
        
    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag if action == 0 else 0.0
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        energy_used = abs(force)

        if not terminated:
            # Old reward function
            # reward = 0.0 if self._sutton_barto_reward else 1.0
            
            # Calculate penalties for pole angle and center position
            pole_angle_penalty = abs(theta) / self.theta_threshold_radians
            center_distance = abs(x)
            if center_distance <= 0.5:
                center_penalty = 0.0
            else:
                center_penalty = (center_distance - 0.5) / (self.x_threshold - 0.5)

            # Base reward calculation for upright and centered state
            base_reward = 1.0 - (pole_angle_penalty + center_penalty)
            base_reward = max(base_reward, 0.0)

            # Energy penalty based on the magnitude of the force applied
            energy_coefficient = 0.01  # Tuning parameter for energy penalty
            energy_penalty = energy_coefficient * energy_used

            # Combine base reward with energy penalty
            reward = max(base_reward - energy_penalty, 0.0)
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0
            
        

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {"energy_used": energy_used}
    