import torch 
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from linear_operator.operators import TriangularLinearOperator
from botorch.utils.safe_math import log_softplus 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softplus_func = torch.nn.Softplus()

def get_expected_log_utility_x_next(
    use_kg,
    acquisition_bsz,
    model,
    x_next,
    kg_samples,
    zs,
    normed_best_f,
    base_samples,
    num_mc_samples_qei,
    use_botorch_stable_log_softplus=False,
):
    if use_kg:
        if acquisition_bsz == 1:
            expected_log_utility_x_next = get_expected_log_utility_knowledge_gradient(
                model=model,
                x_next=x_next, 
                kg_samples=kg_samples, 
                zs=zs,
                normed_best_f=normed_best_f, 
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
            ) 
        else:
            expected_log_utility_x_next = get_q_expected_log_utility_knowledge_gradient(
                model=model,
                x_next=x_next, 
                kg_samples=kg_samples, 
                zs=zs,
                normed_best_f=normed_best_f, 
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
            ) 
    else:
        if acquisition_bsz == 1:
            expected_log_utility_x_next = get_expected_log_utility_ei( 
                model=model, 
                best_f=normed_best_f, 
                x_next=x_next,
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
            )
        else: 
            expected_log_utility_x_next = get_q_expected_log_utility_ei(
                model=model, 
                best_f=normed_best_f, 
                x_next=x_next,
                base_samples=base_samples,
                num_mc_samples=num_mc_samples_qei,
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
            )

    return expected_log_utility_x_next.mean()


def get_q_expected_log_utility_ei(
    model,
    best_f,
    x_next, 
    base_samples,
    num_mc_samples=64,
    use_botorch_stable_log_softplus=False,
):
    # x_next.shape (q, d)
    output = model(x_next) # q-dim multivariate normal 
    # use MC sampling 
    samples = output.rsample(torch.Size([num_mc_samples]), base_samples=base_samples) 
    # compute log utility of each sample 
    if use_botorch_stable_log_softplus:
        log_utilities = log_softplus(samples - best_f)
    else:
        log_utilities = torch.log(softplus_func(samples - best_f)) # (S, q) of utilities for each sample 
    # max over q dimension, mean over s dimension to get final expected_log_utility
    expected_log_utility = log_utilities.amax(-1) # (S,) 
    return expected_log_utility 


def get_expected_log_utility_ei(
    model,
    best_f,
    x_next, # (q,d)
    use_botorch_stable_log_softplus=False,
):
    output = model(x_next) 
    def log_utility(y,):
        # compute log utility based on y and best_f
        if use_botorch_stable_log_softplus:
            log_utility = log_softplus(y - best_f)
        else:
            log_utility = torch.log(softplus_func(y - best_f)) 
        return log_utility.to(device)
    
    ghq = GaussHermiteQuadrature1D()
    ghq = ghq.to(device)
    expected_log_utility = ghq(log_utility, output)
            
    return expected_log_utility

def get_q_expected_log_utility_knowledge_gradient(model, x_next, kg_samples, zs, normed_best_f, use_botorch_stable_log_softplus=False):
    x_next_pred = model(x_next)
    y_samples = x_next_pred.rsample(torch.Size([kg_samples.shape[0]]), base_samples=zs) + x_next_pred.stddev*zs # (num_kg_samples,q) (S,q) 
    chol_factor = model.variational_strategy._cholesky_factor(None) # (M,M)  
    U = model.covar_module(model.variational_strategy.inducing_points, x_next) # (M,q) 
    S = model.covar_module(x_next, x_next, diag=True) # K(x_next, x_next), torch.Size(q), 
    chol_factor_tensor = chol_factor._tensor.tensor # (M,M) 
    chol_factor_tensor_repeated = chol_factor_tensor.repeat(x_next.shape[0], 1, 1,) # (q, M, M)
    L = torch.cat((chol_factor_tensor_repeated, torch.zeros(x_next.shape[0], chol_factor_tensor.shape[-1], 1).to(device)), -1) # (q, M, M+1)
    var_mean = chol_factor @ model.variational_strategy.variational_distribution.mean
    var_mean = var_mean.repeat(x_next.shape[0],1).unsqueeze(-1) # (q,M,1)
    var_mean_repeated = var_mean.repeat(1,1,y_samples.shape[-2]) # (q,M,num_kg_samples)
    y_samples_reshaped = y_samples.reshape(y_samples.shape[-1], y_samples.shape[-2]) # (q,S)
    y_samples_reshaped = y_samples_reshaped.unsqueeze(-2) # (q,1,S)
    var_mean_repeated = torch.cat((var_mean_repeated, y_samples_reshaped), -2) # (q,M+1,num_kg_samples)
    L_12 = chol_factor.solve(U.evaluate_kernel().tensor) # (M,q)
    L_12_mt = L_12.mT.unsqueeze(-1) # (q,M,1)
    schur_complement = S - (L_12_mt * L_12_mt).squeeze(-1).sum(-1) # (q,)
    schur_complement = schur_complement.unsqueeze(-1).unsqueeze(-1) # (q,1,1)
    L_22 = schur_complement.to_dense()**0.5  # (q,1,1)
    L_temp = torch.cat((L_12_mt, L_22), -2) # (q, M+1, 1)
    L_temp_reshaped = L_temp.squeeze().unsqueeze(-2) 
    L = torch.cat((L, L_temp_reshaped), -2) # (q, M+1, M+1)
    L = TriangularLinearOperator(L) 
    alphas = L._transpose_nonbatch().solve(L.solve(var_mean_repeated)) # (q, M+1, S)
    x_next_temp = x_next.unsqueeze(-2) # (q,1,d)
    q_Zs = model.variational_strategy.inducing_points.repeat(x_next.shape[0],1,1) # (q,M,d)
    inducing_points_and_x_next = torch.cat((q_Zs, x_next_temp), -2) # (q, M+1, D)
    constant_mean = model.mean_module.constant
    pred_mean_each_x_sample = model.covar_module(kg_samples, inducing_points_and_x_next) # (q, S, M+1)
    pred_mean_each_x_sample = pred_mean_each_x_sample * alphas.mT 
    pred_mean_each_x_sample = pred_mean_each_x_sample.sum(-1) + constant_mean # (q,S)

    if use_botorch_stable_log_softplus:
        expected_log_utility_kg = log_softplus(pred_mean_each_x_sample - normed_best_f)
    else:
        expected_log_utility_kg = torch.log(softplus_func(pred_mean_each_x_sample - normed_best_f)) # (q, S,)
    expected_log_utility_kg = expected_log_utility_kg.amax(-2) # (S,)
    
    return expected_log_utility_kg 


def get_expected_log_utility_knowledge_gradient(model, x_next, kg_samples, zs, normed_best_f, use_botorch_stable_log_softplus=False):
    x_next_pred = model(x_next)
    y_samples = x_next_pred.mean + x_next_pred.stddev*zs # (num_kg_samples,)
    y_samples = y_samples.unsqueeze(-2) # (1, num_kg_samples) = (1,S)
    chol_factor = model.variational_strategy._cholesky_factor(None) # (M,M)
    U = model.covar_module(model.variational_strategy.inducing_points, x_next) # (M,1)
    S = model.covar_module(x_next, x_next) # K(x_next, x_next)
    chol_factor_tensor = chol_factor._tensor.tensor # (M,M)
    L = torch.cat((chol_factor_tensor, torch.zeros(chol_factor_tensor.shape[-1], 1).to(device)), -1) # (M, M+1)
    var_mean = chol_factor @ model.variational_strategy.variational_distribution.mean
    var_mean = var_mean.unsqueeze(-1) # (M,1)
    var_mean_repeated = var_mean.repeat(1,y_samples.shape[-1]) # (M,num_kg_samples) = (M,S) 
    var_mean_repeated = torch.cat((var_mean_repeated, y_samples)) # (M+1,num_kg_samples) = (M+1, S)
    L_12 = chol_factor.solve(U.evaluate_kernel().tensor) # (M,1)
    schur_complement = S - L_12.mT @ L_12 
    L_22 = schur_complement.to_dense()**0.5 
    L_temp = torch.cat((L_12, L_22), -2)
    L_temp = L_temp.squeeze().unsqueeze(-2) 
    L = torch.cat((L, L_temp), -2) # (M+1, M+1) 
    L = TriangularLinearOperator(L) 
    alphas = L._transpose_nonbatch().solve(L.solve(var_mean_repeated)) # (M+1, S) 
    inducing_points_and_x_next = torch.cat((model.variational_strategy.inducing_points, x_next), -2) # (M+1, D)
    constant_mean = model.mean_module.constant
    pred_mean_each_x_sample = model.covar_module(kg_samples, inducing_points_and_x_next) # (S, M+1) 
    pred_mean_each_x_sample = pred_mean_each_x_sample * alphas.mT 
    pred_mean_each_x_sample = pred_mean_each_x_sample.sum(-1) + constant_mean # (S,) 

    if use_botorch_stable_log_softplus:
        expected_log_utility_kg = log_softplus(pred_mean_each_x_sample - normed_best_f)
    else:
        expected_log_utility_kg = torch.log(softplus_func(pred_mean_each_x_sample - normed_best_f)) # (S,) 

    return expected_log_utility_kg
