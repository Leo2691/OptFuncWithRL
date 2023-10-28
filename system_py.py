import os
import pickle
from datetime import datetime

"""
Script of system or help functions
"""

def prepare_dirs(args):
    """
    Sets the directories for the model and its results
    :param   args: Parsed dict from 'argparse' in the module
    :return: path (to created folder)
    """

    if args.work_mode:
        source = r'Experiments/RL/'
        w_d    = os.path.join(os.getcwd(), source,  args.type_of_surf)
        name_dir = "{}_{}".format(get_time(), args.postfix_name)
        path = os.path.join(w_d, name_dir)

        os.makedirs( path )

        return path

    else:
        source = r'Experiments/RL/__time__'
        path = os.path.join(os.getcwd(), source, get_time())

        os.makedirs( path )

        return path

def get_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

def serialize_obj(path, name, obj):
    path_obj = os.path.join(path, name + '.pickle')
    path_obj = os.path.normpath(path_obj)
    with open(path_obj, 'wb') as outfile:
        pickle.dump(obj, outfile)

def deserilize_obj(path, name):
    path_obj = os.path.join(path, name + '.pickle')
    path_obj = os.path.normpath(path_obj)
    with open(path_obj, 'rb') as infile:
        obj = pickle.load(infile)

    return obj