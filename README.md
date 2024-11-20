# Approximation-Aware Bayesian Optimization

Official implemention of Approximation-Aware Bayesian Optimization https://arxiv.org/abs/2406.04308. 
This repository includes code to run the EULBO method propsed in the paper as well as all beaseline methods on all tasks from the paper.

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Docker
To set up the environment we used to run all tasks for this paper, we recommend using docker. 
You can use the public docker image nmaus/opt (https://hub.docker.com/r/nmaus/opt), or build it yourself using docker/Dockerfile.

### LassoBench DNA Task
The above nmaus/opt docker image has all needed requirements to runn all tasks in the paper except for the LassoBench DNA task. This task requires the following additional steps to setup. These steps can be done after initializing the nmaus/opt docker image which has all other requirements for running AABO. 

```Bash
git clone https://github.com/ksehic/LassoBench.git
cd LassoBench/
pip install -e .
```

## Example Commands

### Run TuRBO with EULBO (E[Log U]) where Utility U is Expected Improvement (EI)
```Bash
cd scripts 
python3 run_bo.py --task_id $TASK_ID --eulbo True - run - done 
```

### Run TuRBO with EULBO (E[Log U]) where Utility U is Knowledge Gradient (KG)
```Bash 
cd scripts 
python3 run_bo.py --task_id $TASK_ID --eulbo True --use_kg True - run - done 
```

### Run TuRBO with regular ELBO 
```Bash 
cd scripts 
python3 run_bo.py --task_id $TASK_ID --eulbo False - run - done 
```

### Run Global BO instead of TuRBO 
Add following arg to any of the run commands above to run with global BO

```Bash
--use_turbo False
```

## Tasks 

Use the following argument to specify the optimization task you'd like to run by providing the string id for the desired task task:

```Bash
--task_id $TASK_ID
```

This code base provides support for the following optimization tasks:

| task_id | Full Task Name     |
|---------|--------------------|
|  hartmann6 | Hartmann 6D     |
|  lunar     | Lunar Lander    |
|  rover     | Rover           |
|  dna       | Lasso DNA       |
|  osmb      | Osimertinib MPO    |
|  fexo      | Fexofenadine MPO   |
|  med1      | Median molecules 1 |
|  med2      | Median molecules 2 |
