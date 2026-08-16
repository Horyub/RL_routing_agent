from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.envs.registration import register, registry



ENV_ID = "RoutingEnv-v4"
HYPERPARAMETERS_PATH = Path(__file__).with_name("hyperparameters.yml")
HYPERPARAMETERS_SET = "routing_env"


def _update_hyperparameters_env_id(env_id: str):
    if not HYPERPARAMETERS_PATH.exists():
        return

    with HYPERPARAMETERS_PATH.open("r", encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file) or {}

    env_hyperparameters = hyperparameters.get(HYPERPARAMETERS_SET)
    if not isinstance(env_hyperparameters, dict):
        env_hyperparameters = {}
        hyperparameters[HYPERPARAMETERS_SET] = env_hyperparameters

    if env_hyperparameters.get("env_id") == env_id:
        return

    env_hyperparameters["env_id"] = env_id

    with HYPERPARAMETERS_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(hyperparameters, file, sort_keys=False)


def register_env(env_id: str = ENV_ID):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point=f"{__name__}:RoutingEnv",
        )

    _update_hyperparameters_env_id(env_id)


class RoutingEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        adjacency_matrix: np.array,
        num_traffic_demands: int = 5,
        ramp_up :bool = True,
        render_mode=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.adjacency_matrix :np.array = adjacency_matrix
        self.n = self.adjacency_matrix.shape[0]
        self.valid_links = np.asarray(self.adjacency_matrix) != 0
        self.number_of_episodes = 0
        
        self.ramp_up = ramp_up
        self.max_number_of_requests = self.n * (self.n -1)
        self.total_requests = min(num_traffic_demands, self.max_number_of_requests)
        self.request_offset = 10
        self.TM :np.array= self._generate_traffic_matrix()
        self.residual_capacity_matrix :np.array = self.adjacency_matrix.copy()

        self.current_valid_actions = np.zeros(self.n, dtype=bool)

        self.source :int = -1
        self.current_node :int = -1
        self.destination :int= -1

        self.completed_requests = 0
        self.traffic_request :np.float64 = 0.0
        
        self.visited = np.zeros(self.n, dtype=np.int64)
        self.hop_count = 0

        self.path = []
   
        self.action_space = gym.spaces.Discrete(self.n)

        self.observation_space = gym.spaces.Dict(
            {
                "current" :gym.spaces.Discrete(self.n),
                "destination" :gym.spaces.Discrete(self.n),
                "visited": gym.spaces.MultiBinary(self.n),
                "current_demand" :gym.spaces.Box(low=0.0, high=0.2, shape=(), dtype=np.float64),
                "link_capacities" :gym.spaces.Box(low=0.0, high=1.0, shape=(self.n,self.n), dtype=np.float64),
                "traffic_matrix" :gym.spaces.Box(low=0.0, high=1.0, shape=(self.n,self.n), dtype=np.float64)
            }
        )


    def render(self):
        info = self._get_info()
        path = " -> ".join(str(node) for node in self.path)

        if self.current_node == self.destination:
            status = "request completed"
        elif info["valid_actions"].size == 0:
            status = "blocked / no valid actions"
        else:
            status = "running"

        used_edges = list(zip(self.path[:-1], self.path[1:]))
        used_capacities = [
            self.residual_capacity_matrix[i, j]
            for i, j in used_edges
        ]

        print()
        print(f"Request:     {self.completed_requests + 1}/{self.total_requests}")
        print(f"Source:      {self.source}")
        print(f"Current:     {self.current_node}")
        print(f"Destination: {self.destination}")
        print(f"Demand:      {self.traffic_request:.4f}")
        print(f"Hop count:   {self.hop_count}")
        print(f"Path:        {path}")
        print(f"Valid moves: {info['valid_actions'].tolist()}")
        print(f"Status:      {status}")

        if used_capacities:
            print(f"Path residual capacities: {[round(c, 4) for c in used_capacities]}")



    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        '''
        Reset the environment for a new episode.
        Generate a new traffic matrix, restore link capacities, and load the first traffic request.
        '''
        self.number_of_episodes +=1 

        if self.number_of_episodes % 1000 == 0 and self.ramp_up:
            self.total_requests = min(self.total_requests+self.request_offset, self.max_number_of_requests)
            if self.total_requests == self.max_number_of_requests:
                self.ramp_up = False

        self.TM = self._generate_traffic_matrix()
        self.residual_capacity_matrix = self.adjacency_matrix.copy()
        self.path = []

        self.source, self.destination = np.argwhere(self.TM != 0)[0]
        self.current_node = self.source

        self.traffic_request :np.float64 = self.TM[self.source, self.destination]
        
        self.visited = np.zeros(self.n, dtype=np.int64)
        self.visited[self.current_node] = 1
        self.path.append(self.source)
        self.hop_count = 0
        self.completed_requests = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info
        



    def step(self, action):
            '''
            Move to the selected node and update the capacity of the used link.
            If the destination is reached, mark the current request as completed and load the next one.
            The episode terminates when all traffic requests are completed.
            If there are no valid actions before termination, the episode is truncated.
            The reward gives a small step penalty, a bonus for completing a request,
            a final residual-capacity bonus when terminated, and a progress-based penalty when truncated.
            '''
            truncated = False
            
            self.hop_count += 1
            self.current_node = action
            self.visited[action] = 1
            self.path.append(action)

            node_i, node_j = self.path[-2], self.path[-1]
            self.residual_capacity_matrix[node_i, node_j] -= self.traffic_request

            path_completed = (self.current_node == self.destination)

            terminated = False
            truncated = False

            if path_completed:
                self.completed_requests += 1
                self.TM[self.source, self.destination] = 0

                terminated = (self.completed_requests == self.total_requests)
                if not terminated:
                    self._load_next_request()


            observation = self._get_obs()
            info = self._get_info()

            no_valid_actions = (info["valid_actions"].size == 0)

            N = self.total_requests

            reward = -0.1 / N

            if path_completed:
                reward += 1.0 / N

            if terminated:
                reward += np.min(self.residual_capacity_matrix[self.valid_links])

            elif no_valid_actions:
                truncated = True

                progress = self.completed_requests / N

                # Failure early is worse than failure late.
                reward += (progress - 1.0)


            if self.render_mode == "human":
                        self.render()

            return observation, reward, terminated, truncated, info





            
    
    def _generate_traffic_matrix(self):
        traffic_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        possible_rows, possible_cols = np.where(~np.eye(self.n, dtype=bool))
        selected = self.np_random.choice(self.max_number_of_requests,size=self.total_requests, replace=False,)
        rows, cols = possible_rows[selected], possible_cols[selected]

        traffic_matrix[rows, cols] = (
            self.np_random.integers(1, 201, size=self.total_requests) / 10000.0
        )
        return traffic_matrix



    def _get_obs(self):
        return {
            "current": self.current_node,
            "destination": self.destination,
            "visited": self.visited,
            "current_demand": np.float64(self.traffic_request),
            "link_capacities": self.residual_capacity_matrix,
            "traffic_matrix" : self.TM
        }

    def _get_info(self):
        action_mask = self._get_action_mask()
        return {
            "hop_count": self.hop_count,
            "valid_actions": np.flatnonzero(action_mask),
            "action_mask": action_mask,
        }

    def _get_action_mask(self):
        '''
        Return the valid next nodes from the current node.
        A node is valid if the link exists, has enough residual capacity,
        and has not already been visited in the current path.
        '''
        action_mask = (
            (self.residual_capacity_matrix[self.current_node] >= self.traffic_request)
            & self.valid_links[self.current_node]
            & (self.visited == 0)
        )
        return action_mask

    def _load_next_request(self):
        self.path = []

        self.source, self.destination = np.argwhere(self.TM != 0)[0]
        self.current_node = self.source

        self.traffic_request :np.float64 = self.TM[self.source, self.destination]
        
        self.visited = np.zeros(self.n, dtype=np.int64)
        self.visited[self.current_node] = 1
        self.path.append(self.source)
        self.hop_count = 0




    

register_env()
