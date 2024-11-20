import torch 
from torch.quasirandom import SobolEngine
from botorch.generation.sampling import MaxPosteriorSampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_kg_samples_and_zs(
    model,
    dim,
    ub,
    lb,
    num_kg_samples,
    acquisition_bsz,
    dtype,
):
    # Use ts to initialize kg_samples 
    thompson_sampling = MaxPosteriorSampling(
        model=model,
        replacement=False,
    ) 
    n_ts_candidates = min(5000, max(2000, 200 * dim)) 
    sobol = SobolEngine(dim, scramble=True) 
    ts_x_cands = sobol.draw(n_ts_candidates).to(device=device).to(dtype=dtype)
    ts_x_cands = ts_x_cands*(ub - lb) + lb 
    with torch.no_grad():
        kg_samples = thompson_sampling(ts_x_cands, num_samples=num_kg_samples ) 
    kg_samples = torch.clone(kg_samples.detach()).requires_grad_(True).to(device=device)
    if acquisition_bsz == 1:
        zs = torch.randn(num_kg_samples, requires_grad=True, device=device) 
    else:
        zs = torch.randn(num_kg_samples, acquisition_bsz, requires_grad=True, device=device)
    return kg_samples, zs 