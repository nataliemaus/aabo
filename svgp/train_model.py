import sys 
sys.path.append("../")
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from torch.autograd import Variable 
from utils.get_turbo_lb_ub import get_turbo_lb_ub
from utils.compute_expected_log_utility import get_expected_log_utility_x_next
from utils.get_kg_samples_and_zs import get_kg_samples_and_zs
from utils.set_inducing_points_with_moss23 import set_inducing_points_with_moss23
import copy 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_model_elbo(
    model,
    train_x,
    train_y,
    mll=None,
    lr=0.01,
    n_epochs=30,
    train_bsz=32,
    grad_clip=1.0,
    train_to_convergence=True, 
    max_allowed_n_failures_improve_loss=10,
    max_allowed_n_epochs=100,
    moss23_baseline=False,
    ppgpr=False,
):
    if mll is None:
        if ppgpr: 
            mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_x.size(-2))
        else:
            mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_x.size(-2))
    model.train()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': lr} ], lr=lr)
    train_bsz = min(len(train_y),train_bsz)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    lowest_loss = torch.inf 
    n_failures_improve_loss = 0
    epochs_trained = 0
    continue_training_condition = True 
    while continue_training_condition:
        total_loss = 0
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = -mll(output, scores.to(device))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        epochs_trained += 1
        if total_loss < lowest_loss:
            lowest_loss = total_loss
        else:
            n_failures_improve_loss += 1
        if train_to_convergence:
            continue_training_condition = n_failures_improve_loss < max_allowed_n_failures_improve_loss
            if epochs_trained > max_allowed_n_epochs:
                continue_training_condition = False 
        else:
            continue_training_condition = epochs_trained < n_epochs
    model.eval()
    if moss23_baseline:
        model = set_inducing_points_with_moss23(model)
    return_dict = {}
    return_dict["model"] = model 
    return return_dict


def update_model_and_generate_candidates_eulbo(
    model,
    train_x,
    train_y,
    lb,
    ub,
    init_x_next=None,
    x_next_lr=0.001,
    mll=None,
    lr=0.01,
    n_epochs=30,
    train_bsz=32,
    normed_best_f=None,
    acquisition_bsz=1,
    grad_clip=2.0,
    max_allowed_n_failures_improve_loss=10,
    max_allowed_n_epochs=100,
    alternate_updates=True,
    num_kg_samples=64, 
    use_kg=False,
    dtype=torch.float64,
    num_mc_samples_qei=64,
    ablation1_fix_indpts_and_hypers=False,
    ablation2_fix_hypers=False,
    use_turbo=True,
    tr_length=None,
    use_botorch_stable_log_softplus=False,
    ppgpr=False,
):
    if use_turbo: 
        assert tr_length is not None 
        lb, ub = get_turbo_lb_ub(
            ub=ub,
            lb=lb,
            X=train_x, 
            Y=train_y,
            tr_length=tr_length,
        )
    if torch.is_tensor(lb):
        lb = lb.to(device)
        ub = ub.to(device)
    torch.autograd.set_detect_anomaly(True) 
    if init_x_next is None:
        init_x_next = torch.rand(acquisition_bsz, train_x.shape[-1], requires_grad=True)*(ub - lb) + lb
    init_x_next = init_x_next.to(device=device)
    train_bsz = min(len(train_y),train_bsz)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    model_state_before_update = copy.deepcopy(model.state_dict())
    n_failures = 0
    success = False 
    while (n_failures < 8) and (not success):
        try:
            model, x_next = eulbo_training_loop(
                dim=train_x.shape[-1],
                num_kg_samples=num_kg_samples,
                model=model,
                mll=mll,
                init_x_next=init_x_next,
                lr=lr,
                x_next_lr=x_next_lr,
                max_allowed_n_epochs=max_allowed_n_epochs,
                train_loader=train_loader,
                n_epochs=n_epochs,
                max_allowed_n_failures_improve_loss=max_allowed_n_failures_improve_loss,
                use_kg=use_kg,
                acquisition_bsz=acquisition_bsz,
                normed_best_f=normed_best_f,
                num_mc_samples_qei=num_mc_samples_qei,
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
                lb=lb,
                ub=ub,
                grad_clip=grad_clip,
                alternate_updates=alternate_updates,
                ablation1_fix_indpts_and_hypers=ablation1_fix_indpts_and_hypers,
                ablation2_fix_hypers=ablation2_fix_hypers,
                ppgpr=ppgpr,
                dtype=dtype,
                n_train=train_x.size(-2),
            )
            success = True
        except Exception as e:
            # decrease lr to stabalize training 
            error_message = e
            n_failures += 1
            lr = lr/10
            x_next_lr = x_next_lr/10
            model.load_state_dict(copy.deepcopy(model_state_before_update))
    if not success:
        assert 0, f"\nFailed to complete EULBO model update due to the following error:\n{error_message}"
    model.eval()
    return_dict = {}
    return_dict["model"] = model 
    return_dict["x_next"] = x_next.detach().cpu()  

    return return_dict


def eulbo_training_loop(
    model,
    mll,
    init_x_next,
    lr,
    x_next_lr,
    max_allowed_n_epochs,
    train_loader,
    n_epochs,
    max_allowed_n_failures_improve_loss,
    use_kg,
    acquisition_bsz,
    normed_best_f,
    num_mc_samples_qei,
    use_botorch_stable_log_softplus,
    lb,
    ub,
    dtype,
    n_train,
    dim,
    num_kg_samples,
    grad_clip=2.0,
    alternate_updates=True,
    ablation1_fix_indpts_and_hypers=False,
    ablation2_fix_hypers=False,
    ppgpr=False,
):
    model.train()
    init_x_next = copy.deepcopy(init_x_next)
    x_next = Variable(init_x_next, requires_grad=True)
    base_samples = torch.randn(num_mc_samples_qei, acquisition_bsz).to(device=device).to(dtype=dtype) 
    if mll is None:
        if ppgpr: 
            mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=n_train)
        else:
            mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=n_train)

    if ablation1_fix_indpts_and_hypers: 
        model_params_to_update = list(model.variational_parameters())
    elif ablation2_fix_hypers: 
        model_params_to_update = list(model.variational_parameters()) + [model.variational_strategy.inducing_points]
    else:
        model_params_to_update = list(model.parameters())
    lowest_loss = torch.inf
    n_failures_improve_loss = 0
    epochs_trained = 0
    continue_training_condition = True 
    if (max_allowed_n_epochs == 0) or (n_epochs == 0):
        continue_training_condition = False 
    currently_training_model = True 
    x_next_optimizer = torch.optim.Adam([{'params': x_next},], lr=x_next_lr)
    model_optimizer = torch.optim.Adam([{'params': model_params_to_update, 'lr':lr} ], lr=lr)
    joint_optimizer = torch.optim.Adam([{'params': x_next,},{'params': model_params_to_update, 'lr':lr} ], lr=lr)
    if use_kg:
        kg_samples, zs = get_kg_samples_and_zs(
            model=model,
            dim=dim,
            ub=ub,
            lb=lb,
            num_kg_samples=num_kg_samples,
            acquisition_bsz=acquisition_bsz,
            dtype=dtype,
        )
    else:
        kg_samples = None
        zs = None 
    while continue_training_condition:
        total_loss = 0
        for (inputs, scores) in train_loader:
            if alternate_updates:
                model_optimizer.zero_grad()
                x_next_optimizer.zero_grad()
            else:
                joint_optimizer.zero_grad()
            output = model(inputs.to(device))
            nelbo = -mll(output, scores.to(device))
            expected_log_utility_x_next = get_expected_log_utility_x_next(
                use_kg=use_kg,
                acquisition_bsz=acquisition_bsz,
                model=model,
                x_next=x_next,
                kg_samples=kg_samples,
                zs=zs,
                normed_best_f=normed_best_f,
                base_samples=base_samples,
                num_mc_samples_qei=num_mc_samples_qei,
                use_botorch_stable_log_softplus=use_botorch_stable_log_softplus,
            )
            loss = nelbo - expected_log_utility_x_next
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                torch.nn.utils.clip_grad_norm_(x_next, max_norm=grad_clip)
            if alternate_updates:
                if currently_training_model: 
                    model_optimizer.step() 
                else:
                    x_next_optimizer.step()
            else:
                joint_optimizer.step()
            with torch.no_grad():   
                x_next[:,:] = x_next.clamp(lb, ub) 
                total_loss += loss.item()
        epochs_trained += 1
        currently_training_model = not currently_training_model 
        if total_loss < lowest_loss:
            lowest_loss = total_loss
        else:
            n_failures_improve_loss += 1
        continue_training_condition = n_failures_improve_loss < max_allowed_n_failures_improve_loss
        if epochs_trained > max_allowed_n_epochs:
            continue_training_condition = False 

    return model, x_next 
