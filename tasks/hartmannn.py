import sys 
sys.path.append("../")
import torch
from tasks.objective import Objective
from botorch.test_functions import Hartmann
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Hartmann6D(Objective):
    """Hartmann function optimization task from
    https://www.sfu.ca/~ssurjano/hart6.html,
    Designed as a minimization task so we multiply by -1
    to obtain a maximization task 
    Using BoTorch implementation: 
    https://botorch.org/v/0.1.2/api/_modules/botorch/test_functions/hartmann6.html
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.neg_hartmann6 = Hartmann(negate=True)

        super().__init__(
            dim=6,
            lb=0,
            ub=1,
            **kwargs,
        )

    def f(self, x):
        x = x.to(device)
        self.num_calls += 1
        y = self.neg_hartmann6(x)
        return y.item()


if __name__ == "__main__":
    obj = Hartmann6D()
    known_optimum = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]).to(dtype=obj.dtype)
    known_optimum = known_optimum.unsqueeze(0)
    best_objective_value = obj(known_optimum)
    print(f"Best possible Hartmann6D val: {best_objective_value}")


