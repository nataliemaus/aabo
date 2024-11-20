import sys 
sys.path.append("../")
import numpy as np
import torch 
from tasks.objective import Objective
import multiprocess as mp
from collections.abc import Iterable 
from tasks.utils.lunar_utils import simulate_lunar_lander


class LunarLanderObjective(Objective):
    ''' Lunar Lander optimization task
        Goal is to find a policy for the Lunar Lander 
        smoothly lands on the moon without crashing, 
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        seed=np.arange(50),
        **kwargs,
    ):
        super().__init__(
            dim=12,
            lb=0.0,
            ub=1.0,
            **kwargs
        ) 
        self.pool = mp.Pool(mp.cpu_count())
        seed = [seed] if not isinstance(seed, Iterable) else seed 
        self.seed = seed 


    def f(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy() 
        x = x.reshape((-1, self.dim)) 
        ns = len(self.seed)
        nx = x.shape[0] 
        x_tiled = np.tile(x, (ns, 1)) 
        seed_rep = np.repeat(self.seed, nx) 
        params = [[xi, si] for xi, si in zip(x_tiled, seed_rep)]
        rewards = np.array(self.pool.map(simulate_lunar_lander, params)).reshape(-1)
        # Compute the average score across the seeds 
        mean_reward = np.mean(rewards, axis=0).squeeze()
        self.num_calls += 1

        return mean_reward

