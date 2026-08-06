import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry


ENV_ID = "ToyEnv-v0"


def register_toy_env(env_id: str = ENV_ID):
    if env_id in registry:
        return

    register(
        id=env_id,
        entry_point=f"{__name__}:ToyEnv",
    )


class ToyEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        max_steps: int | None = None,
        goal_reward: float = 1.0,
        step_penalty: float = -0.05,
        invalid_action_penalty: float = -0.20,
        render_mode=None,
    ):
        super().__init__()

        self.render_mode = render_mode

        self.adjacency = np.asarray(adjacency_matrix) != 0
        self.n = self.adjacency.shape[0]
        self.start_node = np.random.randint(0, self.n)
        

        self.current_node = self.start_node
        
        self.destination_node = np.random.randint(0, self.n)
        while self.destination_node == self.start_node:
            self.destination_node = np.random.randint(0, self.n)

        
        self.visited = np.zeros(self.n, dtype=np.int8)
        self.step_count = 0
        self.path = [self.start_node]

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty

        self.max_steps = (
            max_steps if max_steps is not None
            else 4 * self.n
        )

        self.action_space = gym.spaces.Discrete(self.n)

        self.observation_space = gym.spaces.Dict(
            {
                "current": gym.spaces.Discrete(self.n),
                "destination": gym.spaces.Discrete(self.n),
                "visited": gym.spaces.MultiBinary(self.n),
            }
        )

    def _get_obs(self):
        return {
            "current": self.current_node,
            "destination": self.destination_node,
            "visited": self.visited,
        }

    def _get_info(self):
        action_mask = self.adjacency[self.current_node].astype(np.int8)

        return {
            "step_count": self.step_count,
            "valid_actions": np.flatnonzero(action_mask).tolist(),
            "action_mask": action_mask,
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            path = " -> ".join(str(node) for node in self.path)

            if self.current_node == self.destination_node:
                status = "reached destination"
            elif self.step_count >= self.max_steps:
                status = "max steps reached"
            else:
                status = "running"

            print()
            print(f"Step:        {self.step_count}/{self.max_steps}")
            print(f"Path:        {path}")
            print(f"Current:     {self.current_node}")
            print(f"Destination: {self.destination_node}")
            print(f"Valid moves: {info['valid_actions']}")
            print(f"Status:      {status}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.start_node = np.random.randint(0, self.n)

        self.current_node = self.start_node
        self.step_count = 0
        self.path = [self.start_node]



        self.visited = np.zeros(self.n, dtype=np.int8)
        self.visited[self.start_node] = 1

        self.destination_node = np.random.randint(0, self.n)
        while self.destination_node == self.start_node:
            self.destination_node = np.random.randint(0, self.n)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info



    def step(self, action):
        self.step_count += 1

        transition_is_valid = bool(
            self.adjacency[self.current_node, action]
        )

        if transition_is_valid:
            self.current_node = action
            self.visited[action] = 1
        else:
            # Invalid graph transitions leave the agent where it is.
            reward = self.invalid_action_penalty

        self.path.append(self.current_node)

        terminated = self.current_node == self.destination_node
        truncated = self.step_count >= self.max_steps and not terminated

        if terminated:
            reward = self.goal_reward
        elif transition_is_valid:
            reward = self.step_penalty

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


register_toy_env()
