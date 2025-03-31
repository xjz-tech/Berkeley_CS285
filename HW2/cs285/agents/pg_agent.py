from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
import sys
sys.path.append('/home/xjz/Berkeley_CS285/HW2/cs285')


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)





        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                critic_update_info = self.critic.update(obs, q_values)
                for k, v in critic_update_info.items():
                    if k not in critic_info:
                        critic_info[k] = v
                    else:
                        critic_info[k] += v
            
            # 计算均值
            for k in critic_info:
                critic_info[k] /= self.baseline_gradient_steps

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        q_values = np.empty(len(rewards), dtype=object)  # Initialize an empty NumPy array of dtype=object
    
        if not self.use_reward_to_go:
            # Case 1: trajectory-based PG
            for i, r in enumerate(rewards):
                q_values[i] = np.array(self._discounted_return(rewards=r))  # Store as array
        else:
            # Case 2: reward-to-go PG
            for i, r in enumerate(rewards):
                q_values[i] = np.array(self._discounted_reward_to_go(rewards=r))  # Store as array

        # print("q_values:", q_values)
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values.copy()
        else:
            # TODO: run the critic and use it as a baseline
            obs_tensor = ptu.from_numpy(obs)
            values = ptu.to_numpy(self.critic(obs_tensor))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)
                rewards = np.append(rewards, [0])  # 添加一个虚拟的最后奖励为0

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i]:
                        # 如果是终止状态，则delta = r_t + gamma * V(s_{t+1}) - V(s_t) = r_t - V(s_t)，因为终止状态的下一个状态的价值为0
                        delta = rewards[i] - values[i]
                        advantages[i] = delta
                    else:
                        # 非终止状态，根据GAE公式计算
                        delta = rewards[i] + self.gamma * values[i+1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """

        dis_value = 0
        for n, r in enumerate(rewards):
            q = (self.gamma) ** n * r
            dis_value += q
        return [dis_value for _ in rewards]


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """

        t = len(rewards)  # Correcting the variable name
        q_vals = np.zeros(t)  # Initialize an array of zeros with the correct size

        for i in range(t):
            dis_value = 0
            for j, r in enumerate(rewards[i:]):  # Iterate over rewards starting from index i
                q = (self.gamma) ** j * r  # Correct exponentiation
                dis_value += q
            q_vals[i] = dis_value  # Store the cumulative discounted reward

        return q_vals
