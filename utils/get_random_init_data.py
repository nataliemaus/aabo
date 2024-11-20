import torch 

def get_random_init_data(num_initialization_points, objective):
    ''' randomly initialize num_initialization_points
        total initial data points to kick-off optimization 
        Returns the following:
            init_train_x (a tensor of x's)
            init_train_y (a tensor of scores/y's)
    '''
    lb, ub = objective.lb, objective.ub 
    init_train_x = torch.rand(num_initialization_points, objective.dim)*(ub - lb) + lb
    init_train_y = objective(init_train_x)

    return init_train_x, init_train_y