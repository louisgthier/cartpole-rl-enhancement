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
        
        # Initialize reward weights (default for phase above 25k steps)
        self.alive_weight = 0.25
        self.pole_weight = 0.25
        self.distance_weight = 0.25
        self.energy_weight = 0.25
        
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
        info = {}
    
        if not terminated:
            # Calculate penalties for pole angle and center position
            pole_angle_penalty = abs(theta) / self.theta_threshold_radians
            center_distance = abs(x)
            center_penalty = 0.0 if center_distance <= 0.5 else (center_distance - 0.5) / (self.x_threshold - 0.5)

            # Calculate reward components using environment weights
            distance_reward = self.distance_weight * (1.0 - center_penalty)
            pole_angle_reward = self.pole_weight * (1.0 - pole_angle_penalty)
            alive_reward = self.alive_weight
            energy_coefficient = 0.1  # Adjust as needed
            energy_reward = self.energy_weight * (1.0 - energy_coefficient * energy_used)

            # Ensure rewards are non-negative
            distance_reward = max(distance_reward, 0.0)
            pole_angle_reward = max(pole_angle_reward, 0.0)
            energy_reward = max(energy_reward, 0.0)

            # Sum up total reward from all components
            reward = distance_reward + pole_angle_reward + alive_reward + energy_reward

            # Store sub-rewards/penalties in info
            info = {
                "distance_reward": distance_reward,
                "pole_angle_reward": pole_angle_reward,
                "alive_reward": alive_reward,
                "energy_reward": energy_reward,
                "pole_angle_penalty": -pole_angle_penalty,
                "center_penalty": -center_penalty,
                "energy_penalty": -energy_coefficient * energy_used,
                "energy_used": energy_used
            }
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
            
        info["energy_used"] = energy_used

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info
    