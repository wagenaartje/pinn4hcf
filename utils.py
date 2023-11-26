''' Imports '''
import pickle
import torch
import os

# Retrieves all the model checkpoints
def get_checkpoints (path):
    checkpoints = []

    with open(path, 'rb') as f:
        while True:
            try:
                checkpoints.append(pickle.load(f))
            except BaseException:
                return checkpoints

# Retrieves the last valid model checkpoint
def get_last_params (path, truncate=False):
    checkpoints = get_checkpoints(path)

    for i in range(len(checkpoints) - 1, -1, -1):
        if 'g_params' in checkpoints[i]:
            contains_nan = False
            for key, value in checkpoints[i]['g_params'].items():
                if torch.isnan(value).any():
                    contains_nan = True
                    break
            
            if not contains_nan:
                checkpoint = checkpoints[i]
                break

        checkpoints.pop(i)

    if truncate:
        os.remove(path) 
        with open(path, 'ab') as f:
            for checkpoint in checkpoints:
                pickle.dump(checkpoint, f)

    return checkpoint

# Retrieves a specific checkpoint
def get_checkpoint (path, epoch, truncate=False):
    if epoch == 'latest':
        return get_last_params(path, truncate)
    else:
        checkpoints = get_checkpoints(path)
        for checkpoint in checkpoints:
            if checkpoint['epoch'] == int(epoch):
                return checkpoint