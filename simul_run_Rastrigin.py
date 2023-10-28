import os
import time

import mlflow
import numpy as np
import shutil
import multiprocessing
import subprocess

import system_py as sys_py

import simul_config
import RL_simul
import RL_agents
import RL_evt_opt_func

"""
    Script for forming dict of hyperparameters of RL simulation 
    and launch it in established mode
"""
experiment_name = '0_RL_Rastrigin_test'


hp_sets = [
    {
        'postfix_name'  : '',
        'work_mode'     : False,
        'mlflow_reg'    : False,
        'parallel_mode' : None,
                            # {
                            #     'num_threads'   : 3
                            # },

        'type_of_surf'  : 'Rastrigin',

        'num_agents'    : 1,
        'n_values'      : 100,
        'range_values'  : np.arange(-64, 64, 2/3).tolist(), # np.arange(*range_).tolist(),

        'hidden_dim'    : 256,
        'batch_size'    : 300,

        'optimizer'     : 'Adam',
        'lr'            : 0.001
    }
]

# num_simuls = 5
# hp_sets = [{**hp_sets, 'seed': np.random.randint(10000)} for m in np.arange(num_simuls)]

if __name__ == '__main__':

    threads     = []
    simul_paths = []

    for i, hps in enumerate(hp_sets):
        args, unparsed = simul_config.get_args()
        dict_args = vars(args)

        hps['track_hps'] = list(hps.keys())

        mlflow.set_tracking_uri(args.server_address)
        # tracking results
        if args.mlflow_reg:
            if not hps['work_mode']:
                hps['EXPERIMENT_NAME'] = '__time__'       ## test experiment
            else:
                hps['EXPERIMENT_NAME'] = experiment_name  ## real experiment folder

            #Getting MLFlow experoment id and run id
            try:
                hps['EXPERIMENT_ID'] = mlflow.create_experiment(hps['EXPERIMENT_NAME'])
            except:
                hps['EXPERIMENT_ID'] = mlflow.set_experiment(hps['EXPERIMENT_NAME']).experiment_id
            hps['RUN_ID'] = df = mlflow.search_runs(hps['EXPERIMENT_ID']).shape[0]
            hps['RUN_NAME'] = f"run_{hps['RUN_ID']}"

        # updating of hyperparameters dict by hps
        dict_args.update(hps)

        # preparation of directory and taking new actual path
        simul_paths.append(sys_py.prepare_dirs(args))
        dict_args['simul_path'] = simul_paths[-1]

        sys_py.serialize_obj(simul_paths[-1],   'dict_args', dict_args)
        shutil.copy2(simul_config.__file__,     simul_paths[-1])
        shutil.copy2(RL_simul.__file__,         simul_paths[-1])
        shutil.copy2(RL_agents.__file__,        simul_paths[-1])
        shutil.copy2(RL_evt_opt_func.__file__,  simul_paths[-1])

        time.sleep(1)

        ## threads running
        # threads.append(
        #     multiprocessing.Process(target=RL_agent.main, args=(simul_paths[-1],))
        # )
        # threads[-1].start()

        # usual serail running | for debugging in pycharm
        RL_simul.main(simul_paths[-1])

        # For Mac OS (does not work)--------------------------------------------------------------------------
        # subprocess.Popen(['/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal', '-e',
        #                   'python', 'RL_simul.py', simul_paths[-1]],
        #                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
        # process = subprocess.Popen(['/path/to/venv/bin/python3', 'script.py'],

        # for Windows ----------------------------------------------------------------------------------------
        # subprocess.Popen('cmd.exe/ c start python ' + RL_simul.__file__ + ' ' + simul_paths[-1], start_new_session = True)


