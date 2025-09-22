import sys 
sys.path.append("../")
import torch
import fire 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import copy 
import gpytorch
from svgp.model import GPModel
from svgp.generate_candidates import generate_batch
from svgp.train_model import (
    update_model_elbo, 
    update_model_and_generate_candidates_eulbo,
)
from utils.create_wandb_tracker import create_wandb_tracker
from utils.set_seed import set_seed 
from utils.get_random_init_data import get_random_init_data
from utils.turbo import TurboState, update_state
from utils.set_inducing_points_with_moss23 import get_optimal_inducing_points
# for exact gp baseline: 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# for specific tasks
from tasks.hartmannn import Hartmann6D
from tasks.rover import RoverObjective
try:
    from tasks.lunar import LunarLanderObjective
    successful_lunar_import = True
except:
    print("Warning: failed to import LunarLanderObjective, current environment does not support needed imports for lunar lander task")
    successful_lunar_import = False 
try:
    from tasks.lasso_dna import LassoDNA
    successful_lasso_dna_import = True 
except:
    print("Warning: failed to import LassoDNA Objective, current environment does not support needed imports for Lasso DNA task")
    successful_lasso_dna_import = False 
try:
    from tasks.guacamol_objective import GuacamolObjective
except:
    print("Warning: failed to import GuacamolObjective, current environment does not support needed imports for guacamol tasks")

task_id_to_objective = {}
task_id_to_objective['hartmann6'] = Hartmann6D
if successful_lunar_import:
    task_id_to_objective['lunar'] = LunarLanderObjective 
task_id_to_objective['rover'] = RoverObjective
if successful_lasso_dna_import: 
    task_id_to_objective['dna'] = LassoDNA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimize(object):
    """
    Run Approximation Aware Bayesian Optimization (AABO)
    Args:
        task_id: String id for optimization task in task_id_to_objective dict 
        seed: Random seed to be set. If None, no particular random seed is set
        wandb_entity: Username for your wandb account for wandb logging
        wandb_project_name: Name of wandb project where results will be logged, if none specified, will use default f"run-aabo-{task_id}"
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        bsz: Acquisition batch size
        train_bsz: batch size used for model training/updates
        num_initialization_points: Number of initial random data points used to kick off optimization
        lr: Learning rate for model updates
        x_next_lr: Learning rate for x next updates with EULBO method 
        acq_func: Acquisition function used for warm-starting model, must be either ei, logei, or ts (logei--> Log Expected Imporvement, ei-->Expected Imporvement, ts-->Thompson Sampling)
        n_update_epochs: Number of epochs to update the model for on each optimization step
        n_inducing_pts: Number of inducing points for GP
        grad_clip: clip the gradeint at this value during model training 
        eulbo: If True, use EULBO for model training and canidate selection (AABO), otherwise use the standard ELBO (i.e. standard BO baselines).
        use_turbo: If True, use trust region BO, used for higher-dim tasks in the paper 
        use_kg: If True, use EULBO-KG. Otherwise, use EULBO-EI 
        exact_gp_baseline: If True, instead of AABO run baseline of vanilla BO with exact GP 
        ablation1_fix_indpts_and_hypers: If True, run AABO ablation from paper where inducing points and hyperparams remain fixed (not udated by EULBO)
        ablation2_fix_hypers: If True, run AABO ablation from paper where hyperparams remain fixed (not udated by EULBO)
        moss23_baseline: If True, instead of AABO run the moss et al. 2023 paper method baseline (use inducing point selection method of every iteration of optimization)
        inducing_pt_init_w_moss23: If True, use moss et al. 2023 paper method to initialize inducing points at the start of optimizaiton 
        normalize_ys: If True, normalize objective values for training (recommended, typical when using GP models)
        max_allowed_n_failures_improve_loss: We train model until the loss fails to improve for this many epochs
        max_allowed_n_epochs: Although we train to convergence, we also cap the number of epochs to this max allowed value
        n_warm_start_epochs: Number of epochs used to warm start the GP model with standard ELBO before beginning training with EULBO
        alternate_eulbo_updates: If true, we alternate updates of model and x_next when training with EULBO (imporves training convergence and stability)
        update_on_n_pts: Update model on this many data points on each iteration.
        num_kg_samples: number of samples used to compute log utility with KG 
        num_mc_samples_qei: number of MC samples used to ocmpute log utility with aEI 
        float_dtype_as_int: specify integer either 32 or 64, dictates whether to use torch.float32 or torch.float64 
        use_botorch_stable_log_softplus: if True, use botorch new implementation of log softplus (https://botorch.org/api/_modules/botorch/utils/safe_math.html#log_softplus)
        verbose: if True, print optimization progress updates 
        ppgpr:  if True, use PPGPR instead of SVGP 
        run_id: Optional string run id. Only use is for wandb logging to identify a specific run
        init_with_guacamol: if true, initialize Guacamol tasks with guacamol data (otherwise initialize them randomly)
    """
    def __init__(
        self,
        task_id: str="rover",
        seed: int=None,
        wandb_entity: str="",
        wandb_project_name: str="",
        max_n_oracle_calls: int=20_000,
        bsz: int=20,
        train_bsz: int=32,
        num_initialization_points: int=100,
        lr: float=0.01,
        x_next_lr: float=0.001,
        acq_func: str="ei",
        n_update_epochs: int=5,
        n_inducing_pts: int=100,
        grad_clip: float=2.0,
        eulbo: bool=True,
        use_turbo=True,
        use_kg=False,
        exact_gp_baseline=False,
        ablation1_fix_indpts_and_hypers=False,
        ablation2_fix_hypers=False,
        moss23_baseline=False,
        inducing_pt_init_w_moss23=True,
        normalize_ys=True,
        max_allowed_n_failures_improve_loss: int=3,
        max_allowed_n_epochs: int=30, 
        n_warm_start_epochs: int=10,
        alternate_eulbo_updates=True,
        update_on_n_pts=100,
        num_kg_samples=64,
        num_mc_samples_qei=64,
        float_dtype_as_int=64,
        use_botorch_stable_log_softplus=False,
        verbose=True,
        ppgpr=False,
        init_with_guacamol=True,
        run_id="",
    ):
        if float_dtype_as_int == 32:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)
        elif float_dtype_as_int == 64:
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            assert 0, f"float_dtype_as_int must be one of: [32, 64], instead got {float_dtype_as_int}"

        if ablation1_fix_indpts_and_hypers:
            assert eulbo
            assert not ablation2_fix_hypers
        if ablation2_fix_hypers:
            assert eulbo
            assert not ablation1_fix_indpts_and_hypers
        
        if moss23_baseline:
            assert not eulbo
            assert not exact_gp_baseline
            assert inducing_pt_init_w_moss23

        if eulbo:
            assert not exact_gp_baseline

        if exact_gp_baseline:
            assert not eulbo
        
        # log all args to wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        wandb_config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}

        self.run_id = run_id
        self.ppgpr = ppgpr
        self.use_botorch_stable_log_softplus = use_botorch_stable_log_softplus
        self.init_training_complete = False
        self.n_warm_start_epochs = n_warm_start_epochs
        self.normalize_ys = normalize_ys
        self.ablation1_fix_indpts_and_hypers = ablation1_fix_indpts_and_hypers
        self.ablation2_fix_hypers = ablation2_fix_hypers
        self.inducing_pt_init_w_moss23 = inducing_pt_init_w_moss23
        self.moss23_baseline = moss23_baseline
        self.exact_gp_baseline = exact_gp_baseline
        self.use_turbo = use_turbo
        self.update_on_n_pts = update_on_n_pts
        self.verbose = verbose
        self.x_next_lr = x_next_lr
        self.alternate_eulbo_updates = alternate_eulbo_updates
        self.max_allowed_n_failures_improve_loss = max_allowed_n_failures_improve_loss
        self.max_allowed_n_epochs = max_allowed_n_epochs
        self.eulbo = eulbo
        self.max_n_oracle_calls=max_n_oracle_calls
        self.n_inducing_pts=n_inducing_pts
        self.lr=lr
        self.n_update_epochs=n_update_epochs
        self.train_bsz=train_bsz
        self.grad_clip=grad_clip
        self.bsz=bsz
        self.acq_func=acq_func
        self.num_kg_samples = num_kg_samples
        self.use_kg = use_kg
        self.num_mc_samples_qei = num_mc_samples_qei
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        # start wandb tracker
        if not wandb_project_name:
            wandb_project_name = f"run-aabo-{task_id}"
        self.tracker = create_wandb_tracker(
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            config_dict=wandb_config_dict,
        )
        signal.signal(signal.SIGINT, self.handler)

        # Define objective and get initialization data
        if task_id in task_id_to_objective:
            self.objective = task_id_to_objective[task_id](dtype=self.dtype)
            # get random init training data
            self.train_x, self.train_y = get_random_init_data(
                num_initialization_points=num_initialization_points,
                objective=self.objective,
            )
        else: # if task_id is not in dict, assume specific gaucamol objective task
            self.objective = GuacamolObjective(guacamol_task_id=task_id, dtype=self.dtype)
            if init_with_guacamol:
                # load guacamol data for initialization 
                df = pd.read_csv("../tasks/utils/selfies_vae/train_ys.csv")
                self.train_y = torch.from_numpy(df[task_id].values).float()
                self.train_x = torch.load("../tasks/utils/selfies_vae/train_zs.pt")
                self.train_x = self.train_x[0:num_initialization_points]
                self.train_y = self.train_y[0:num_initialization_points]
                self.train_y, top_k_idxs = torch.topk(self.train_y, min(self.update_on_n_pts, len(self.train_y)))
                self.train_x = self.train_x[top_k_idxs]
                self.train_y = self.train_y.unsqueeze(-1)
                self.train_y = self.train_y.to(dtype=self.dtype)
                self.train_x = self.train_x.to(dtype=self.dtype)
            else:
                # get random init training data
                self.train_x, self.train_y = get_random_init_data(
                    num_initialization_points=num_initialization_points,
                    objective=self.objective,
                )


        if self.verbose:
            print("train x shape:", self.train_x.shape)
            print("train y shape:", self.train_y.shape)

        # get normalized train y
        self.train_y_mean = self.train_y.mean()
        self.train_y_std = self.train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1
        if self.normalize_ys:
            self.normed_train_y = (self.train_y - self.train_y_mean) / self.train_y_std
        else:
            self.normed_train_y = self.train_y

        # Initialize turbo state
        self.tr_state = TurboState(
            dim=self.train_x.shape[-1],
            batch_size=self.bsz,
            best_value=self.train_y.max().item(),
        )
        # Initialize GP model
        if not self.exact_gp_baseline:
            # get inducing points
            if len(self.train_x) >= self.n_inducing_pts:
                inducing_points = self.train_x[0:self.n_inducing_pts,:]
            else:
                n_extra_ind_pts = self.n_inducing_pts - len(self.train_x)
                extra_ind_pts = torch.rand(n_extra_ind_pts, self.objective.dim)*(self.objective.ub - self.objective.lb) + self.objective.lb
                inducing_points = torch.cat((self.train_x, extra_ind_pts), -2)
            self.initial_inducing_points = copy.deepcopy(inducing_points) 
            # Define approximate GP model
            learn_inducing_locations = True
            if self.moss23_baseline:
                learn_inducing_locations = False
            self.model = GPModel(
                inducing_points=inducing_points.to(self.device),
                likelihood=gpytorch.likelihoods.GaussianLikelihood().to(self.device),
                learn_inducing_locations=learn_inducing_locations,
            ).to(self.device)
            if self.inducing_pt_init_w_moss23:
                optimal_inducing_points = get_optimal_inducing_points(
                    model=self.model,
                    prev_inducing_points=inducing_points,
                )
                self.model = GPModel(
                    inducing_points=optimal_inducing_points,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood().to(self.device),
                    learn_inducing_locations=learn_inducing_locations,
                ).to(self.device)
                self.initial_inducing_points = copy.deepcopy(optimal_inducing_points)


    def grab_data_for_update(self,):
        if not self.init_training_complete:
            x_update_on = self.train_x
            normed_y_update_on = self.normed_train_y.squeeze()
            self.init_training_complete = True
        else:
            x_update_on = self.train_x[-self.update_on_n_pts:]
            normed_y_update_on = self.normed_train_y.squeeze()[-self.update_on_n_pts:]

        return x_update_on, normed_y_update_on


    def run(self):
        ''' Main optimization loop
        '''
        while self.objective.num_calls < self.max_n_oracle_calls:
            # Update wandb with optimization progress
            best_score_found = self.train_y.max().item()
            n_calls_ = self.objective.num_calls
            if self.verbose:
                print(f"After {n_calls_} oracle calls, Best reward = {best_score_found}")
            # Log data to wandb on each loop
            dict_log = {
                "best_found":best_score_found,
                "n_oracle_calls":n_calls_,
            }
            self.tracker.log(dict_log)
            # Normalize train ys
            if self.normalize_ys:
                self.normed_train_y = (self.train_y - self.train_y_mean) / self.train_y_std
            else:
                self.normed_train_y = self.train_y
            # Update model on data collected
            if self.exact_gp_baseline:
                # re-init exact gp model
                self.model = SingleTaskGP(
                    self.train_x,
                    self.normed_train_y,
                    covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood().to(self.device),
                )
                exact_gp_mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
                # fit model to data
                fit_gpytorch_mll(exact_gp_mll)
            else:
                if self.eulbo:
                    n_epochs_elbo = self.n_warm_start_epochs
                    train_to_convergence_elbo = False
                else:
                    n_epochs_elbo = self.n_update_epochs
                    train_to_convergence_elbo = True
                x_update_on, normed_y_update_on = self.grab_data_for_update()
                update_model_dict = update_model_elbo(
                        model=self.model,
                        train_x=x_update_on,
                        train_y=normed_y_update_on,
                        lr=self.lr,
                        n_epochs=n_epochs_elbo,
                        train_bsz=self.train_bsz,
                        grad_clip=self.grad_clip,
                        train_to_convergence=train_to_convergence_elbo,
                        max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                        max_allowed_n_epochs=self.max_allowed_n_epochs,
                        moss23_baseline=self.moss23_baseline,
                        ppgpr=self.ppgpr,
                    )
                self.model = update_model_dict["model"]
            # Generate a batch of candidates 
            x_next = generate_batch(
                model=self.model,
                X=self.train_x,  
                Y=self.normed_train_y,
                batch_size=self.bsz,
                acqf=self.acq_func,
                device=self.device,
                absolute_bounds=(self.objective.lb, self.objective.ub),
                use_turbo=self.use_turbo,
                tr_length=self.tr_state.length,
                dtype=self.dtype,
            )
            # If using EULBO, use above model update and candidate generaiton as warm start
            if self.eulbo:
                lb = self.objective.lb
                ub = self.objective.ub
                update_model_dict = update_model_and_generate_candidates_eulbo(
                    model=self.model,
                    train_x=x_update_on,
                    train_y=normed_y_update_on,
                    lb=lb,
                    ub=ub,
                    lr=self.lr,
                    n_epochs=self.n_update_epochs,
                    train_bsz=self.train_bsz,
                    grad_clip=self.grad_clip,
                    normed_best_f=self.normed_train_y.max(),
                    acquisition_bsz=self.bsz,
                    max_allowed_n_failures_improve_loss=self.max_allowed_n_failures_improve_loss,
                    max_allowed_n_epochs=self.max_allowed_n_epochs,
                    init_x_next=x_next,
                    x_next_lr=self.x_next_lr,
                    alternate_updates=self.alternate_eulbo_updates,
                    num_kg_samples=self.num_kg_samples,
                    use_kg=self.use_kg,
                    dtype=self.dtype,
                    num_mc_samples_qei=self.num_mc_samples_qei,
                    ablation1_fix_indpts_and_hypers=self.ablation1_fix_indpts_and_hypers,
                    ablation2_fix_hypers=self.ablation2_fix_hypers,
                    use_turbo=self.use_turbo,
                    tr_length=self.tr_state.length,
                    use_botorch_stable_log_softplus=self.use_botorch_stable_log_softplus,
                    ppgpr=self.ppgpr,
                )
                self.model = update_model_dict["model"]
                x_next = update_model_dict["x_next"]

            # Evaluate candidates
            y_next = self.objective(x_next)

            # Update data
            self.train_x = torch.cat((self.train_x, x_next), dim=-2)
            self.train_y = torch.cat((self.train_y, y_next), dim=-2)

            # if running TuRBO, update trust region state
            if self.use_turbo:
                self.tr_state = update_state(
                    state=self.tr_state,
                    Y_next=y_next,
                )
                if self.tr_state.restart_triggered:
                    self.tr_state = TurboState(
                        dim=self.train_x.shape[-1],
                        batch_size=self.bsz,
                        best_value=self.train_y.max().item(),
                    )
        self.tracker.finish()
        return self

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we terminate wandb tracker
        print("Ctrl-c hass been pressed, terminating wandb tracker...")
        self.tracker.finish()
        msg = "tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)


if __name__ == "__main__":
    fire.Fire(Optimize)
