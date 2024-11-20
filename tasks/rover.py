import torch 
import sys 
sys.path.append("../")
from tasks.objective import Objective
from tasks.utils.rover_utils import create_large_domain, ConstantOffsetFn


class RoverObjective(Objective):
    ''' Rover optimization task
        Goal is to find a policy for the Rover which
        results in a trajectory that moves the rover from
        start point to end point while avoiding the obstacles,
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        dim=60,
        **kwargs,
    ):
        assert dim % 2 == 0
        lb = -0.5 * 4 / dim 
        ub = 4 / dim 
        self.domain = create_large_domain(n_points=dim // 2)
        f_max=5.0 
        self.oracle = ConstantOffsetFn(self.domain, f_max)
        self.tkwargs={"dtype": torch.double}

        super().__init__(
            dim=dim,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 


    def f(self, x):
        self.num_calls += 1
        reward = torch.tensor(self.oracle(x.cpu().numpy())).to(**self.tkwargs) 
        return reward.item()

if __name__ == "__main__":
    N = 10
    obj = RoverObjective()
    x_next = torch.rand(N, obj.dim)*(obj.ub - obj.lb) + obj.lb
    y_next = obj(x_next)
    print(
        "Scores:", y_next, y_next.shape
    )
