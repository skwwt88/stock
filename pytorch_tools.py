import random
import torch
import numpy as np
import os
import glob 

def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def model_name(name, index, score):
     return "{0}_{1:03d}_{2:.6f}.pkl".format(name, index, score)

def model_exist(name, model_folder):
     files = glob.glob(os.path.join(model_folder, "{}*.pkl".format(name)))
     return True if files and len(files) > 0 else False

def find_best_model_file(name, model_folder, use_max = True):
     files = glob.glob(os.path.join(model_folder, "{}*.pkl".format(name)))
     files.sort()
     return files[-1] if use_max else files[0]

def clean_models(name, model_folder):
     for file in glob.glob(os.path.join(model_folder, "{}*.pkl".format(name))):
          os.remove(file)

def save_model(model, name, index, model_folder, score):
     if not os.path.exists(model_folder):
          os.makedirs(model_folder)

     name = model_name(name, index, score)
     torch.save(model.state_dict(), os.path.join(model_folder, name))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, name, model_folder, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = name
        self.model_folder = model_folder
        self.saved_model_count = 0
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        save_model(model, self.name, self.saved_model_count, self.model_folder, val_loss)
        self.val_loss_min = val_loss
        self.saved_model_count += 1