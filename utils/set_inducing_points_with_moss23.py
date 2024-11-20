import sys
sys.path.append("../")
import torch
from utils.mossetal_inducing_pts_init import GreedyImprovementReduction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimal_inducing_points(model, prev_inducing_points):
    greedy_imp_reduction = GreedyImprovementReduction(
        model=model,
        maximize=True,
    )
    optimal_inducing_points = greedy_imp_reduction.allocate_inducing_points(
        inputs=prev_inducing_points.to(device),
        covar_module=model.covar_module,
        num_inducing=prev_inducing_points.shape[0],
        input_batch_shape=1,
    )
    return optimal_inducing_points


def set_inducing_points_with_moss23(model):
    prev_inducing_pts = model.variational_strategy.inducing_points
    optimal_inducing_points = get_optimal_inducing_points(
        model=model,
        prev_inducing_points=prev_inducing_pts,
    )
    n_prev_inducing_pts = prev_inducing_pts.shape[0]
    n_new_inducing_pts = optimal_inducing_points.shape[0]
    if n_prev_inducing_pts > n_new_inducing_pts:
        n_prev_needed = n_prev_inducing_pts - n_new_inducing_pts
        optimal_inducing_points = torch.cat((
            prev_inducing_pts[0:n_prev_needed],
            optimal_inducing_points,
        ))
    model.variational_strategy.inducing_points = optimal_inducing_points
    return model
