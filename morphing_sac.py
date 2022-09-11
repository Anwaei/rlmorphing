import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from copy import deepcopy
import itertools
import time
from utils_sp.logx import EpochLogger
from hysterisis_utils import args_setting


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.act_half_dim = int(act_dim/2)
        self.net = mlp([obs_dim + 1] + list(hidden_sizes), activation, activation)  # +1 for ref_now
        self.mu_layer = nn.Linear(hidden_sizes[-1], self.act_half_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], self.act_half_dim)
        self.act_limit = act_limit
        self.act_previous_layer = nn.Identity()

    def forward(self, obs, act_pre, ref_now, deterministic=False, with_logprob=True):
        net_out = self.net(torch.cat(obs, ref_now))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        act_p = self.act_previous_layer(act_pre[self.act_half_dim:])
        pi_action = torch.cat((act_p, pi_action))

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, act_pre, ref_now, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, act_pre, ref_now, deterministic, False)
            return a.numpy()


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
            index[i] = np.random.randint(low=0, high=len(self.conditions))
            ref[i*K_each:(i+1)*K_each] = self.conditions[index[i]]
        ref = torch.as_tensor(ref)
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

        return observation, info, ref[0]

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

        ref_now = self.ref[self.step_count]

        return observation, reward, terminated, info, ref_now


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ref_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, ref_now):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ref_buf[self.ptr] = ref_now
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     ref=self.ref_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def sac(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d, r_n = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['ref']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2, a, r_n)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o, a, r_n = data['obs'], data['act'], data['ref']
        pi, logp_pi = ac.pi(o, a, r_n)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, a, r_n, deterministic=False):
        return ac.act(torch.as_tensor(o, a, r_n, dtype=torch.float32),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, r_n = test_env.reset()
            a = torch.zeros(act_dim)
            ep_ret, ep_len = 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = get_action(o, a, r_n, True)
                o, r, d, r_n, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, d, r_n = env.reset()
    ep_ret, ep_len = 0, 0
    a = torch.zeros(act_dim)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o, a, r_n)
        else:
            a1 = a[act_dim/2:]
            a2 = env.action_space.sample()[0:act_dim]
            a = torch.cat((a1, a2))

        # Step the env
        o2, r, d, r_n, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, r_n)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, d, r_n = env.reset()
            ep_ret, ep_len = 0, 0
            a = torch.zeros(act_dim)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


def generate_env(args):
    return AirfoilEnv(args_airfoil=args)


if __name__ == '__main__':
    args_airfoil = args_setting()
    env = AirfoilEnv(args_airfoil=args_airfoil)
    sac(env_fn=lambda: generate_env(args_airfoil))


