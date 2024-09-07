import torch
import numpy as np
import random

# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)

### model loads specific parameters (i.e., par) from pretrained_model 
def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)