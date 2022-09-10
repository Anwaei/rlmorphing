import gym
from gym import spaces
import numpy as np
from hysterisis_utils import args_setting


class AirfoilEnv(gym.Env):
    metadata = {}

    def __init__(self, args_airfoil, render_mode=None, time_final=20, dt=0.1):
        self._unpack(args_airfoil)
        self.observation_space = spaces.Dict(
            {
                "gamma_p": spaces.Box(low=0, high=1),
                "T_p": spaces.Box(low=0, high=np.inf),
                "gamma": spaces.Box(low=0, high=1),
                "T": spaces.Box(low=0, high=np.inf),
            }
        )
        self.action_space = spaces.Dict(
            {
                "v_p": spaces.Box(low=0, high=10),
                "v": spaces.Box(low=0, high=10)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.time_final = time_final
        self.K = int(time_final/dt)
        self.step_count = 0
        self.reset()

    def _unpack(self, args):
        self.H = args['H']
        self.ctl = args['ctl']
        self.ctr = args['ctr']
        self.b = args['b']
        self.w = args['w']
        self.cs = args['cs']

        self.mw = args['mw']
        self.Aw = args['Aw']
        self.cw = args['cw']
        self.Rw = args['Rw']
        self.Tf = args['Tf']
        self.hw = args['hw']
        self.h2 = args['h2']
        self.K1 = 1 / (self.mw * self.cw * self.Rw)
        self.K2 = -self.hw * self.Aw / (self.mw * self.cw)

        self.dt = args['dt']

        self.conditions = args['conditions']

    def generate_ref(self):
        nc = 10
        K_each = int(self.K / nc)
        ref = np.zeros(self.K)
        index = np.zeros(nc)
        for i in range(nc):
            index[i] = np.randint(0, len(self.conditions))
            ref[i*K_each:(i+1)*K_each] = self.conditions[index[i]]
        return ref, index

    def _get_obs(self):
        return {"gamma_p": self._gamma, "T_p": self._T, "gamma": self._gamma, "T": self._T}

    def _get_info(self):
        return {"reward": (self._gamma - self.ref[self.step_count])**2,
                "ref_current": self.ref[self.step_count]}

    def reset(self, *args, seed = None, return_info = False, options = None):
        super().reset(seed=seed)
        self._gamma_p = 0
        self._gamma = 0
        self._T_p = self.np_random.random()*10 + 20
        self._T = self._T_p

        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        ref, index = self.generate_ref()
        self.ref = ref

        return observation, info

    @staticmethod
    def _func_sgn(x):
        return 1 if x >= 0 else 0

    def _func_sigma(self, T, v):
        return self.K1*v**2 + self.K2*(T-self.Tf)

    def _func_T(self, T_p, v_p):
        return T_p + self.dt * self._func_sigma(T_p, v_p)

    def _func_fl(self, T, h):
        return h/2*np.tanh((T-self.ctl)*self.b) + self.w*(T-(self.ctl+self.ctr)/2) + h/2 + self.cs

    def _func_flinv(self, T, gamma):
        return 2 * (gamma - self.w*(T-(self.ctl+self.ctr)/2) - self.cs) / (np.tanh((T-self.ctl)*self.b) + 1)

    def _func_fr(self, T, h):
        return h/2*np.tanh((T-self.ctr)*self.b) + self.w*(T-(self.ctl+self.ctr)/2) + self.H - h/2 + self.cs

    def _func_frinv(self, T, gamma):
        return 2 * (gamma - self.w*(T-(self.ctl+self.ctr)/2) - self.H - self.cs) / (np.tanh((T-self.ctr)*self.b) - 1)

    def _func_f(self, h, T, T_p, v_p):
        kesi = self._func_sgn(self._func_sigma(T_p, v_p))
        return self._func_fr(T, h) if kesi == 1 else self._func_fl(T, h)

    def _func_finv(self, gamma, T, T_p, v_p):
        kesi = self._func_sgn(self._func_sigma(T_p, v_p))
        return self._func_frinv(T, gamma) if kesi == 1 else self._func_flinv(T, gamma)

    def _func_gl(self, T, h_p):
        return (h_p * (np.tanh((T-self.ctr)*self.b)-1) + 2*self.H) / (np.tanh((T-self.ctl)*self.b)+1)

    def _func_gr(self, T, h_p):
        return (h_p * (np.tanh((T-self.ctl)*self.b)+1) - 2*self.H) / (np.tanh((T-self.ctr)*self.b)-1)

    def _func_gamma(self, T, gamma_p, T_p, v_p, T_pp, v_pp):
        kesi2 = self._func_sgn(self._func_sigma(T_p, v_p) * self._func_sigma(T_pp, v_pp))
        h_p = self._func_finv(gamma_p, T_p, T_pp, v_pp)
        if kesi2 == 1:
            h = h_p
        else:
            kesi = self._func_sgn(self._func_sigma(T_p, v_p))
            h = self._func_gr(T, h_p) if kesi == 1 else self._func_gl(T, h_p)
        gamma = self._func_f(h, T, T_p, v_p)
        return gamma

    def _func_state_transition(self, action):
        v_pp = action[0]
        v_p = action[1]
        T_pp = self._T_p
        T_p = self._T
        gamma_p = self._gamma

        T = self._func_T(T_p, v_p)
        gamma = self._func_gamma(T, gamma_p, T_p, v_p, T_pp, v_pp)

        self._T_p = T_p
        self._T = T
        self._gamma_p = gamma_p
        self._gamma = gamma
        return None

    def step(self, action):
        self._func_state_transition(action)
        observation = self._get_obs()

        reward = (self._gamma - self.ref[self.step_count])**2

        self.step_count += 1
        terminated = self.step_count == self.K-1

        info = self._get_info()

        return observation, reward, terminated, info


if __name__ == '__main__':
    args_airfoil = args_setting()
    env = AirfoilEnv(args_airfoil=args_airfoil)


