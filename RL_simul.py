import argparse
import os
import sys
import time
import warnings

import mlflow
import numpy as np
import pandas as pd

# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from multiprocessing import Pool

from RL_agents import PNG_LSTM
from RL_agents import get_action
from RL_agents import distributor_acts
from RL_evt_opt_func import Rastrigin_env
from RL_evt_opt_func import Rosenbrock_env


import system_py as sys_py

"""
    Script for research global (multy-dementional) optimization based on RL-methods
    for traditional optimization test functions:
        - Rastrigin  function;
        - Rosenbrock function;
        
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def main(path):
    dict_args   = sys_py.deserilize_obj(path, 'dict_args')
    args        = argparse.Namespace(**dict_args)

    print("Enviroment  : ", args.type_of_surf, " function")
    print("Number of actions (values): ", args.n_values)
    print("Search space: [{:05.2f}, {:05.2f}, {:05.2f}] ".format( np.min(args.range_values), np.max(args.range_values),
                                                                        (args.range_values[1] -  args.range_values[0]) ))
    print("Search range: ", str( len(args.range_values) ))

    # Fix random seed
    np.random.seed(args.seed)

    print('Agent has created--------------------------------------------------------------------------')
    print(path)

    # """Tensorboard object---------------------------------------------------------------------------------"""
    # writer = SummaryWriter()

    """mlflow registration params--------------------------------------------------------------------"""
    if args.mlflow_reg:
        mlflow.set_tracking_uri(args.server_address)

        if not args.__dict__.get('run', False):
            # create new mlflow tracking session
            # try:
            args.run = mlflow.start_run(experiment_id=args.EXPERIMENT_ID, run_name=args.RUN_NAME)
            [mlflow.log_param(k, getattr(args, k)) if len(str(getattr(args, k))) < 500
            else mlflow.log_param(k, str(getattr(args, k))[0:499])
            for k in args.track_hps]
            # except:
            #     pass

        else:
            # continue existing mlflow tracking session
            # try:
            args.run = mlflow.start_run(run_id=args.run.info.run_id)
            # except:
            #     pass

    # fixed start state
    size_range_values   = len(args.range_values)
    start_state         = 0.01 * np.random.random((1, args.n_values, size_range_values))
    args.range_values   = np.array(args.range_values)

    # creation reinforce agent
    net = PNG_LSTM( size_range_values, size_range_values, hidden_dim=args.hidden_dim)

    print("Agent configuration: ")
    print(net)

    # Optimizer setting coeffs and params
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # creation df for holding stat about exploration of environment
    df_path = os.path.join(path, 'df.csv')
    try:
        df = pd.read_csv(df_path)
    except:
        column_names = ['actions', 'score']
        df = pd.DataFrame(columns=column_names)
        df.to_csv(df_path, index=False)

    ##  Collection of batch size and Exploration of the inviriment
    # batch_actions = []
    # batch_values  = []
    # batch_rewards = []

    best_score      = 10000000000
    best_values     = []
    best_actions    = []

    for i in np.arange(args.learning_epochs):

        batch_actions = []
        batch_values  = []
        batch_rewards = []
        # start = time.time()
        while len(batch_actions) < args.batch_size:

            ## TODO: Make get_action parralel
            if not args.parallel_mode:
                # start = time.time()
                acts, vals  = get_action(net, start_state, args.range_values)
                acts_str    = ' '.join(str(e) for e in acts)
                # end = time.time()
                # print('Time of get_action : ', end-start)

                ## Calling enviroment event---------------------------------------------------------------------------------
                # start = time.time()
                if   args.type_of_surf == 'Rastrigin':
                     score = Rastrigin_env(vals)
                elif args.type_of_surf == 'Rosenbrock':
                     score = Rosenbrock_env(vals)

                # end = time.time()
                # print('Time of Orcle calling: ', end - start)

                ## TODO: checking of received vals in common database (for expensive oracles)

                batch_actions.append(acts)
                batch_values.append(vals)
                batch_rewards.append(score)


            else:
                num_threads = args.parallel_mode['num_threads']
                p = Pool(num_threads)
                params = []
                [params.append([net, start_state, args.range_values, args.seed + num]) for num in np.arange(num_threads)]
                res = p.map(distributor_acts, params)

                [(batch_actions.append(set_v[0]),
                  batch_values.append(set_v[1]),
                  batch_rewards.append(Rastrigin_env(set_v[1])))
                 for set_v in res]

                score = min(batch_rewards)
                index_min = np.argmin(batch_rewards)
                acts = batch_actions[index_min]
                vals = batch_values[index_min]
                p.close()


            if score < best_score:
                best_score, best_actions, best_values = score, acts, vals



        # end = time.time()
        # print('Time of batch collection : ', end - start)

        batch_actions.append(best_actions)
        batch_rewards.append(best_score)

        """RL training-------------------------------------------------------------------------------------------"""
        ## collection of statistic of rewards for one batch
        batch_rewards = np.array(batch_rewards)

        mean_rewards = float(np.mean(batch_rewards))
        best_reward  = float(np.min(batch_rewards))
        print("%d iterations | mean reward: %6.2f | best reward in batch %6.2f" % (i, mean_rewards, best_reward))
        ## TODO: mlflow writing mean_rewards, best_reward

        ## normalization of batch reward
        batch_rewards = np.array(batch_rewards)
        # batch_rewards = batch_rewards - batch_rewards.mean()
        ## nother option of normalization
        batch_rewards = 100 * np.array(batch_rewards)
        batch_rewards = np.arctan(batch_rewards-batch_rewards.mean())

        optimizer.zero_grad()
        size_st = len(batch_actions)
        start_state_t   = torch.FloatTensor(start_state)
        start_state_t   = torch.tile(start_state_t, (size_st, 1, 1))

        batch_actions_t = torch.LongTensor(batch_actions)#.view(size_st, args.n_values, 1)
        batch_rewards_t = torch.FloatTensor(batch_rewards)

        logits_v, (_, _) = net(start_state_t)
        log_prob_v       = F.log_softmax(logits_v, dim=2)
        mask = torch.nn.functional.one_hot(batch_actions_t, num_classes=size_range_values)

        # k - aditional scale of gradient
        k = -1
        log_prob_actions_v  = k * batch_rewards_t * torch.masked_select(log_prob_v, (mask == 1)).view(size_st, -1).T
        loss_policy_v       = -log_prob_actions_v.mean()

        prob_v          = F.softmax(logits_v, dim=2)
        entropy_v       = -(prob_v * log_prob_v).sum(dim=2).mean()
        entropy_loss_v  = -args.entropy_beta * entropy_v

        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        grad_max    = 0.0
        grad_mean   = 0.0
        grad_count  = 0

        for p in net.parameters():
            grad_max    = max(grad_max, p.grad.abs().max().item())
            grad_mean   += (p.grad ** 2).mean().sqrt().item()
            grad_count  += 1

        # for tensorboard--------------------------------------------------------------------------------------
        # writer.add_scalar("Entropy",        entropy_v.item(),       i)
        # writer.add_scalar("loss_entropy",   entropy_loss_v.item(),  i)
        # writer.add_scalar("loss_polisy",    loss_policy_v.item(),   i)
        # writer.add_scalar("loss_total",     loss_v.item(),          i)
        # writer.add_scalar("grad_l2",        grad_mean / grad_count, i)
        # writer.add_scalar("grad_max",       grad_max,               i)

        # for MLFlow-------------------------------------------------------------------------------------------
        if args.mlflow_reg:
            mlflow.log_metric(key="Mean_Rewards", value=mean_rewards, step=i)
            mlflow.log_metric(key="Best_Reward",  value=best_reward,  step=i)

            mlflow.log_metric(key="Entropy",        value=entropy_v.item(),         step=i)
            mlflow.log_metric(key="loss_entropy",   value=entropy_loss_v.item(),    step=i)
            mlflow.log_metric(key="loss_polisy",    value=loss_policy_v.item(),     step=i)
            mlflow.log_metric(key="loss_total",     value=loss_v.item(),            step=i)
            mlflow.log_metric(key="grad_l2",        value=grad_mean / grad_count,   step=i)
            mlflow.log_metric(key="grad_max",       value=grad_max,                 step=i)




if __name__ == '__main__':
    # Turning off warnings
    warnings.filterwarnings('ignore', '.*do not.*',)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')

    warnings.warn('Do not show this message')

    path = sys.argv[1]

    main(path)



