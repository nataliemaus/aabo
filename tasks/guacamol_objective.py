import sys 
sys.path.append("../")
import numpy as np 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from rdkit import Chem
import selfies as sf 
# import guacamol 
from guacamol import standard_benchmarks 
from tasks.utils.selfies_vae.data import SELFIESDataset
from tasks.utils.selfies_vae.model_positional_unbounded import InfoTransformerVAE

med1 = standard_benchmarks.median_camphor_menthol() #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil() #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings() # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO' 
siga = standard_benchmarks.sitagliptin_replacement() #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula() # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop() # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop() # Scaffold Hop'
rano = standard_benchmarks.ranolazine_mpo() #'Ranolazine MPO' 
fexo = standard_benchmarks.hard_fexofenadine() # 'Fexofenadine MPO'... 'make fexofenadine less greasy'

guacamol_objs = {
    "med1":med1,"pdop":pdop, 
    "adip":adip, "rano":rano, 
    "osmb":osmb, "siga":siga, 
    "zale":zale, "valt":valt,
    "med2":med2,"dhop":dhop, 
    "shop":shop, 'fexo':fexo
} 

class GuacamolObjective:
    """Guacamol optimization tasks
    https://github.com/BenevolentAI/guacamol,
    Using LS-BO with SELFIES VAE from LOL-BO 
    """
    def __init__(
        self,
        guacamol_task_id,
        dim=256,
        num_calls=0,
        dtype=torch.float32,
        lb=-8, # based on forwarding 20k guacamol molecules through vae and seeing min of zs -6.3683
        ub=8, # based on forwarding 20k guacamol molecules through vae and seeing max of zs 7.2140
        path_to_vae_statedict="../tasks/utils/selfies_vae/selfies-vae-state-dict.pt",
        max_string_length=128,
        **kwargs,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim 
        self.dim = dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        self.guacamol_obj_func = guacamol_objs[guacamol_task_id].objective
        self.initialize_vae()
    
    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), 
            each item in enumerable type must be a float tensor of shape (dim,) 
            (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by passing each x in xs into f(x).
        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        xs = xs.to(device)
        smiles_list = self.vae_decode(z=xs)
        ys = []
        for smile in smiles_list:
            y = self.smile_to_guacamole_score(smile=smile)
            if y is None:
                ys.append(-0.01)
            else:
                ys.append(y)
            self.num_calls += 1

        return torch.tensor(ys).to(dtype=self.dtype).unsqueeze(-1)


    def smile_to_guacamole_score(self, smile):
        if smile is None or len(smile)==0:
            return None
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        score = self.guacamol_obj_func.score(smile)
        if score is None:
            return None
        if score < 0:
            return None
        return score 

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True) 
        self.vae = self.vae.to(device)
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).to(dtype=self.dtype)
        z.to(device)
        self.vae.eval()
        self.vae.to(device)
        # sample molecular string form VAE decoder
        with torch.no_grad():
            sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)

        return decoded_smiles


if __name__ == "__main__":
    obj = GuacamolObjective(guacamol_task_id="rano")
    x = torch.randn(12, 256).to(dtype=obj.dtype)
    y = obj(x)
    print(f"y: {y}")

