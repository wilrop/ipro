import numpy as np
import torch as th


class SumTree:
    """SumTree with fixed size."""

    def __init__(self, max_size):
        """Initialize the SumTree.
        Args:
            max_size: Maximum size of the SumTree
        """
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    def sample(self, batch_size):
        """Batch binary search through sum tree. Sample a priority between 0 and the max priority and then search the tree for the corresponding index.
        Args:
            batch_size: Number of indices to sample
        Returns:
            indices: Indices of the sampled nodes
        """
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index

    def set(self, node_index, new_priority):
        """Set the priority of node at node_index to new_priority.
        Args:
            node_index: Index of the node to update
            new_priority: New priority of the node
        """
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        """Batched version of set.
        Args:
            node_index: Index of the nodes to update
            new_priority: New priorities of the nodes
        """
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


class PrioritizedAccruedRewardReplayBuffer:
    """Replay buffer with accrued rewards stored."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            rew_dim=1,
            max_size=100000,
            obs_dtype=np.float32,
            action_dtype=np.float32,
            max_priority=1e-5,
    ):
        """Initialize the Replay Buffer.

        Args:
            obs_shape: Shape of the observations
            action_shape: Shape of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
            max_priority: Minimum priority of the buffer
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.timesteps = np.zeros((max_size, 1), dtype=np.float32)

        self.tree = SumTree(max_size)
        self.max_priority = max_priority
        self.start_max_priority = max_priority

    def reset(self):
        """Reset the buffer."""
        self.ptr, self.size = 0, 0
        self.max_priority = self.start_max_priority
        self.tree = SumTree(self.max_size)

    def reset_priorities(self):
        """Reset the priorities of the buffer."""
        if self.size > 0:
            self.max_priority = self.start_max_priority
            self.update_priorities(np.arange(self.size), np.full(self.size, self.max_priority))

    def add(self, obs, accrued_reward, action, reward, next_obs, done, timestep, priority=None):
        """Add a new experience to memory.

        Args:
            obs: Observation
            accrued_reward: Accrued reward
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
            priority: Priority of the new experience
        """
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.timesteps[self.ptr] = np.array(timestep).copy()

        self.tree.set(self.ptr, self.max_priority if priority is None else priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, to_tensor=False, device=None):
        """Sample a batch of experiences.

        Args:
            batch_size: Number of elements to sample
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        idxes = self.tree.sample(batch_size)

        experience_tuples = (
            self.obs[idxes],
            self.accrued_rewards[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.next_obs[idxes],
            self.dones[idxes],
            self.timesteps[idxes]
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples)) + (idxes,)
        else:
            return experience_tuples + (idxes,)

    def sample_obs_acc_rews(self, batch_size, to_tensor=False, device=None):
        """Sample a batch of observations from the buffer.
        Args:
            batch_size: Number of observations to sample
            to_tensor: Whether to convert the batch to a tensor
            device: Device to move the tensor to
        Returns:
            batch: Batch of observations
        """
        idxes = self.tree.sample(batch_size)
        experience_tuples = (
            self.obs[idxes],
            self.accrued_rewards[idxes],
            self.timesteps[idxes],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples)) + (idxes,)
        else:
            return experience_tuples + (idxes,)

    def update_priorities(self, idxes, priorities):
        """Update the priorities of the experiences at idxes.
        Args:
            idxes: Indexes of the experiences to update
            priorities: New priorities of the experiences
        """
        self.max_priority = max(self.max_priority, priorities.max())
        self.tree.batch_set(idxes, priorities)

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """Returns the whole buffer (with a specified maximum number of samples).

        Args:
            max_samples: the number of samples to return, if not specified, returns the full buffer (ordered!)
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
            self.timesteps[inds]
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size


class AccruedRewardReplayBuffer:
    """Replay buffer with accrued rewards stored."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            rew_dim=1,
            max_size=100000,
            obs_dtype=np.float32,
            action_dtype=np.float32,
    ):
        """Initialize the Replay Buffer.

        Args:
            obs_shape: Shape of the observations
            action_shape:  Shape of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.timesteps = np.zeros((max_size, 1), dtype=obs_dtype)

    def add(self, obs, accrued_reward, action, reward, next_obs, done, timestep):
        """Add a new experience to memory.

        Args:
            obs: Observation
            accrued_reward: Accrued reward
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.timesteps[self.ptr] = timestep

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences.

        Args:
            batch_size: Number of elements to sample
            replace: Whether to sample with replacement or not
            use_cer: Whether to use CER or not
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
            self.timesteps[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def reset(self):
        """Cleanup the buffer."""
        self.size, self.ptr = 0, 0

    def update_priorities(self, idxes, priorities):
        pass

    def sample_obs_acc_rews(self, batch_size, to_tensor=False, device=None):
        pass

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """Returns the whole buffer (with a specified maximum number of samples).

        Args:
            max_samples: the number of samples to return, if not specified, returns the full buffer (ordered!)
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
            self.timesteps[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size


class RolloutBuffer:
    """Rollout buffer."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            rew_dim=1,
            max_size=100,
            obs_dtype=np.float32,
            action_dtype=np.float32,
            aug_obs=False,
    ):
        """Initialize the Rollout Buffer.

        Args:
            obs_shape: Shape of the observations
            action_shape:  Shape of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        if aug_obs:
            obs_shape = (obs_shape[0] + rew_dim,)

        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, size=1):
        """Add a new experience to memory.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        self.obs[self.ptr:self.ptr + size] = np.array(obs).copy()
        self.next_obs[self.ptr:self.ptr + size] = np.array(next_obs).copy()
        self.actions[self.ptr:self.ptr + size] = np.array(action).copy()
        self.rewards[self.ptr:self.ptr + size] = np.array(reward).copy()
        self.dones[self.ptr:self.ptr + size] = np.array(done).copy()

        self.ptr = (self.ptr + size) % self.max_size
        self.size = min(self.size + size, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences.

        Args:
            batch_size: Number of elements to sample
            replace: Whether to sample with replacement or not
            use_cer: Whether to use CER or not
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def reset(self):
        """Cleanup the buffer."""
        self.size, self.ptr = 0, 0

    def update_priorities(self, idxes, priorities):
        pass

    def sample_obs_acc_rews(self, batch_size, to_tensor=False, device=None):
        pass

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """Returns the whole buffer (with a specified maximum number of samples).

        Args:
            max_samples: the number of samples to return, if not specified, returns the full buffer (ordered!)
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        experience_tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size