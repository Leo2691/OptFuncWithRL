import numpy as np
import argparse

arg_list = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg

## Common --------------------------------------------------------------------------------------------------------------
misc_args = add_argument_group('Misc')
misc_args.add_argument('--work_mode',       type = bool, default = False, help='True:  Parallel console multi-agent working mode.'
                                                                                             'False: Debugging one-agent mode')
misc_args.add_argument('--postfix_name',    type = str, default = '')
misc_args.add_argument('--server_address',  type = str, default = 'http://127.0.0.1:5001')
misc_args.add_argument('--mlflow_reg',      type = bool,default = True)


## Simulation-----------------------------------------------------------------------------------------------------------
simul = add_argument_group('Simulation')

simul.add_argument('--num_agents',      type = int, default = 1)
simul.add_argument('--learning_epochs', type = int, default = 100000)

## Enviroment-----------------------------------------------------------------------------------------------------------
env = add_argument_group('Enviroment')

## https://en.wikipedia.org/wiki/Test_functions_for_optimization
env.add_argument('--type_of_surf', type = str, default = 'Rastrigin', choices = ['Rastrigin', 'Rosenbrock']) ## ... list of test optimization functions
# env.add_argument('--size_dim'    , type = int, default = 2, help = 'Number of dimentions in the test fuction')
# size_dim = env._actions[-1].default

## Agent----------------------------------------------------------------------------------------------------------------
agent = add_argument_group('Agent')

agent.add_argument('--n_values',        type = int, default = 2, help = 'Number of predicted values') # the same as 'size_dim' in Enviroment (number of actions)
range_ = [np.round(-agent._actions[-1].default / 2), np.round(agent._actions[-1].default / 2) + 1, 1]
agent.add_argument('--range_values',    type = list,  default = np.arange(*range_).tolist(),  help = 'Range of predicted values')
agent.add_argument('--entropy_beta',    type = float, default = 0.00,                         help = 'Weight of entropy_beta part of loss. Range: [0.05, 0.1]')
agent.add_argument('--batch_size',      type = int,   default = 200,                          help = 'Size of batch for training net')
agent.add_argument('--hidden_dim',      type = int,   default = 12,                           help = 'Size of hidden state of RNN net')


## Optimization--------------------------------------------------------------------------------------------------------
opt = add_argument_group('Optimization')
opt.add_argument('--optimizer', type = str,   default = 'Adam', choices = ['SGD', 'Adam', 'BFGS', 'LBFGS'])
opt.add_argument('--lr',        type = float, default = 0.01, help = 'Learning rate of optimizer')
opt.add_argument('--seed',      type = int,   default = 123,  help = 'Fixed random seed')

def get_args():
    """
    Parsing all arguments above, whixh mostly correspond to the
    hyperparameters of model and its adaptatation
    :return: args, unparsed
    """

    args, unparsed = parser.parse_known_args()

    return args, unparsed

