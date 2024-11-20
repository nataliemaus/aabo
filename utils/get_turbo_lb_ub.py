import copy 
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_turbo_lb_ub(ub, lb, X, Y, tr_length):
    if lb is None:
        lb = X.min().item() 
    if ub is None:
        ub = X.max().item()
    x_center = copy.deepcopy(X[Y.argmax(), :]) 
    weights = torch.ones_like(x_center)
    weights = weights * (ub - lb)
    tr_lb = torch.clamp(x_center - weights * tr_length / 2.0, lb, ub) 
    tr_ub = torch.clamp(x_center + weights * tr_length / 2.0, lb, ub) 
    return tr_lb.to(device), tr_ub.to(device) 
