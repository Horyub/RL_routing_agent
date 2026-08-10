import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry



ENV_ID = "RoutingEnv-v0"


def register_env(env_id: str = ENV_ID):
    if env_id in registry:
        return

    register(
        id=env_id,
        entry_point=f"{__name__}:RoutingEnv",
    )


class RoutingEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        adjacency_matrix: np.array,
        goal_reward: float = 1.0,
        step_penalty: float = 0.0,
        invalid_action_penalty: float = -0.5*12,
        render_mode=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.adjacency_matrix :np.array = adjacency_matrix
        self.n = self.adjacency_matrix.shape[0]
        self.valid_links = np.asarray(self.adjacency_matrix) != 0

        self.TM :np.array= self._generate_traffic_matrix()
        self.residual_capacity_matrix :np.array = self.adjacency_matrix.copy()

        self.current_valid_actions = np.zeros(self.n, dtype=bool)

        self.source :int = -1
        self.current_node :int = -1
        self.destination :int= -1

        self.total_requests = np.count_nonzero(self.TM)
        self.completed_requests = 0
        self.traffic_request :np.float64 = 0.0
        
        self.visited = np.zeros(self.n, dtype=np.int64)
        self.hop_count = 0

        self.path = []

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty

        
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
        Initialize source, destination and traffic request based on TM, then update visited, current pos and path
        '''
        self.TM = self._generate_traffic_matrix()
        self.residual_capacity_matrix = self.adjacency_matrix.copy()
        self.path = []

        self.source, self.destination = np.argwhere(self.TM != 0)[0]
        self.current_node = self.source

        self.total_requests = np.count_nonzero(self.TM)
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
            A single step is equal to one hop, when it is performed the residual capacity
            of the used link is updated.
            Reward is given when a path is completed (src, ..., dest) based on the number of hops:
                r = 0.1 * #hops/#requests
            zero if path isn't completed,
            and a negative reward if it is truncated. 
            When terminated, it gives a reward computed as: 
                r = min(residual capacity)
            The episode is terminated if all traffic requests are processed
            After doing the action, check if there is a new available action, if not, truncate the episode. 

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
                completed_hops = self.hop_count
                self.completed_requests += 1
                self.TM[self.source, self.destination] = 0

                terminated = (self.completed_requests == self.total_requests)
                if not terminated:
                    self._load_next_request()


            observation = self._get_obs()
            info = self._get_info()

            no_valid_actions = (info["valid_actions"].size == 0)

            if no_valid_actions:
                truncated = True
                reward = self.invalid_action_penalty

            elif terminated:
                reward = np.min(self.residual_capacity_matrix[self.valid_links])

            elif path_completed:
                reward = - 0.1 * completed_hops / self.total_requests
            else: 
                reward = self.step_penalty

            if self.render_mode == "human":
                        self.render()

            return observation, reward, terminated, truncated, info





            
    
    def _generate_traffic_matrix(self):
        traffic_matrix = self.np_random.integers(0, 2001, size=(self.n, self.n)) / 10000.0
        np.fill_diagonal(traffic_matrix, 0.0)
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
        Given the current traffic request (tf), extract the row corresponding to the current node i from the residual capacity matrix (RC)
        and check whether:
            tf <= RC[i][j], for every j
        and if the node was already visited, and then make and AND between the two conditions 
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
