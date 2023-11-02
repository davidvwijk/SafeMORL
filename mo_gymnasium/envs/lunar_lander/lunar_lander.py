import math
from scipy.optimize import minimize
import time

# from copy import deepcopy
import copy
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    LEG_DOWN,
    MAIN_ENGINE_POWER,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    VIEWPORT_H,
    VIEWPORT_W,
    LunarLander,
)


class Dynamics:
    def get_next_state(self, action):
        """
        Obtain the states of the lander upon application of action.
        Should not modify the lander states.
        """

        # Get original lander states for later assignment
        box2d_states = [
            self.lander.position[0],
            self.lander.position[1],
            self.lander.linearVelocity[0],
            self.lander.linearVelocity[1],
            self.lander.angle,
            self.lander.angularVelocity,
            self.legs[0].ground_contact,
            self.legs[1].ground_contact,
        ]
        box2d_extra = [self.lander.awake, self.lander.active]

        # TODO: Need to figure out a way to make sure contacts are transferred as well
        # if len(self.lander.contacts) == 0:
        #     self.lander.contacts.append("2")

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action[0] > 0.0:
            # Main engine
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if np.abs(action[1]) > 0.5:
            # Orientation engines
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)

            assert s_power >= 0.5 and s_power <= 1.0

            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        pos = self.lander.position
        vel = self.lander.linearVelocity
        states = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        # Re-assign the states in the environment
        self.lander.position[0] = box2d_states[0]
        self.lander.position[1] = box2d_states[1]
        self.lander.linearVelocity[0] = box2d_states[2]
        self.lander.linearVelocity[1] = box2d_states[3]
        self.lander.angle = box2d_states[4]
        self.lander.angularVelocity = box2d_states[5]
        self.legs[0].ground_contact = box2d_states[6]
        self.legs[1].ground_contact = box2d_states[7]
        self.lander.awake = box2d_extra[0]
        self.lander.active = box2d_extra[1]

        # Euler integration for next position:
        next_position = [states[0] + states[2] * (1 / FPS), states[1] + states[3] * (1 / FPS)]  # [x, y]

        return next_position


class Constraint(Dynamics):
    """
    Constraints to enforce with ASIF.

    """

    def h_x_position(self, x_coord, x_lim):
        h = x_lim**2 - x_coord**2
        return h

    def h_x_orientation(self, theta, theta_lim):  # TODO: Orientation constraint? Or maybe angular velocity constraint
        h = 1
        return h

    def discrete_grad_h_position(self, x_pos, u, limits, delta_t):
        next_state = self.get_next_state(u)[0]
        state = x_pos

        dgrad_h = (self.h_x_position(next_state, limits) - self.h_x_position(state, limits)) / delta_t
        return dgrad_h

    def alpha(self, x):  # TODO: tune alpha
        """
        Strengthening function to relax constraint from the barrier.

        """
        return 15 * x


class ASIF(Constraint):
    """
    Active Set Invariance Filter class monitors the desired agent control and
    tries to find a safe control that is minimally invasive.

    """

    def obj_fun(self, u, u_des):
        """
        Objective function to minimize norm between actual and desired control.

        """
        return np.linalg.norm(u - u_des) / 10

    def DCBF_constraint_fun_pos(self, u, x_pos, limits, delta_t):
        dgrad_h = self.discrete_grad_h_position(x_pos, u, limits, delta_t)
        h_x_curr = self.h_x_position(x_pos, limits)
        return dgrad_h + self.alpha(h_x_curr)

    def rta(self, u_des):
        x_lim = 0.5  # [m] # TODO: Get a good value for this
        delta_t = 1.0 / FPS  # TODO: Make sure this is right
        x_pos = self.lander.position[0]
        x_pos = (x_pos - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)  # Convert to [m]
        if abs(x_pos) > 0.45:
            a = 1
        constraint_list = []
        constraint_list.append(
            {
                "type": "ineq",
                "fun": self.DCBF_constraint_fun_pos,
                "args": (x_pos, x_lim, delta_t),
            }
        )
        constraints = tuple(constraint_list)
        bnds = ((-1, 1), (-1, 1))

        # try:
        # u_0 = u_des  # Initial guess for the solver

        opt = {"disp": True, "maxiter": 203}
        u_0 = np.zeros(2)
        tic = time.perf_counter()
        result = minimize(
            self.obj_fun,
            u_0,
            constraints=constraints,
            method="SLSQP",
            bounds=bnds,
            args=u_des,
            options=opt,
        )
        toc = time.perf_counter()
        print(toc - tic)
        u_act = result.x
        print(self.DCBF_constraint_fun_pos(u_act, x_pos, x_lim, delta_t))
        # except:
        #     u_act = u_des
        #     # if self.verbose:
        #     print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.01:
            intervening = True
            print(self.get_next_state(u_act))
            print(self.get_next_state(u_des))
        else:
            intervening = False
        print("RTA intervened:", intervening)

        return u_act, self.get_next_state(u_act)


class MOLunarLander(LunarLander, ASIF):  # no need for EzPickle, it's already in LunarLander
    """
    ## Description
    Multi-objective version of the LunarLander environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/box2d/lunar_lander/) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: -100 if crash, +100 if lands successfully
    - 1: Shaping reward
    - 2: Fuel cost (main engine)
    - 3: Fuel cost (side engine)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Result reward, shaping reward, main engine cost, side engine cost
        self.reward_space = spaces.Box(
            low=np.array([-100, -np.inf, -1, -1]),
            high=np.array([100, np.inf, 0, 0]),
            shape=(4,),
            dtype=np.float32,
        )
        self.reward_dim = 4
        self.rta_active = True
        print("-----------------------------")
        print("ASIF Active:", self.rta_active)
        print("-----------------------------")

    def step(self, action):
        assert self.lander is not None

        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + (math.sin(math.pi * 0.01 * self.wind_idx))) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(math.sin(0.02 * self.torque_idx) + (math.sin(math.pi * 0.01 * self.torque_idx))) * (
                self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
            # asif = ASIF()
            # action = asif.rta(action)
            if self.rta_active:
                action, predicted_x_pos = self.rta(action)
        else:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        pos_og = [
            (self.lander.position[0] - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (self.lander.position[1] - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        ]

        m_power = 0.0
        if self.continuous and action[0] > 0.0:
            # Main engine
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if self.continuous and np.abs(action[1]) > 0.5:
            # Orientation engines
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)

            assert s_power >= 0.5 and s_power <= 1.0

            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        try:
            print(state[0] / predicted_x_pos[0])
        except:
            a = 1
        assert len(state) == 8

        # print(state[0], state[1])
        reward = 0
        vector_reward = np.zeros(4, dtype=np.float32)
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
            vector_reward[1] = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        vector_reward[2] = -m_power
        reward -= s_power * 0.03
        vector_reward[3] = -s_power

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            vector_reward[0] = -100
        if not self.lander.awake:
            terminated = True
            reward = +100
            vector_reward[0] = +100

        if self.render_mode == "human":
            self.render()

        return np.array(state, dtype=np.float32), vector_reward, terminated, False, {"original_reward": reward}
