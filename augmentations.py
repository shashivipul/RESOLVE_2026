import numpy as np
import torch

class Config_augmentation:
    def __init__(self, jitter_ratio=0.05, jitter_scale_ratio=1.1, max_seg=5, keepratio=0.9, thres = 50):
        self.jitter_ratio = jitter_ratio
        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.keepratio = keepratio
        self.thres = thres

        self.augmentation = {
            'jitter_ratio': jitter_ratio,
            'jitter_scale_ratio': jitter_scale_ratio,
            'max_seg': max_seg,
            'keepratio': keepratio,
            'thres' : thres
        }
config = Config_augmentation() 


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b



def random_frequency_perturbation_one_hot(data):
    choice = np.random.randint(0, 2)  
    one_hot = one_hot_encoding(np.array([choice])) 

    if one_hot[0, 0] == 1:
        return add_frequency(data, pertub_ratio=0.1)  
    else:
        return remove_frequency(data, pertub_ratio=0.1)  


    

def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug

def DataTransform_TD(sample, config):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance."""
    aug = jitter(sample, config.augmentation.jitter_ratio)
    return aug


###############################################################################

def DataTransform_TD_bank(sample, config):
    """
    Apply a mixture of data augmentations (jitter, scaling, permutation, masking) 
    to a subset of randomly selected signals.

    Args:
    - sample (np.ndarray): Shape [features, timesteps], a single sample.
    - config (ConfigAugmentation): Configuration with augmentation settings.
    - thres (float): Percentage of signals to be perturbed (0-100).

    Returns:
    - np.ndarray: Augmented sample of shape [features, timesteps].
    """

    sample = np.array(sample)  
    thres = config.thres
    num_features = sample.shape[0]  
    num_perturb = max(1, int(num_features * thres / 100))  
    selected_indices = np.random.choice(num_features, num_perturb, replace=False)
    li_onehot = np.random.randint(0, 2, size=(num_perturb, 4)) 

    while np.any(li_onehot.sum(axis=1) == 0):  
        li_onehot[np.where(li_onehot.sum(axis=1) == 0)] = np.random.randint(0, 2, size=(1, 4))


    aug_1 = jitter(sample[selected_indices], config.jitter_ratio) * li_onehot[:, 0][:, None]  
    aug_2 = scaling(sample[selected_indices], config.jitter_scale_ratio) * li_onehot[:, 1][:, None]
    aug_3 = permutation(sample[selected_indices], max_segments=config.max_seg) * li_onehot[:, 2][:, None]
    aug_4 = masking(sample[selected_indices], keepratio=config.keepratio) * li_onehot[:, 3][:, None]


    aug_selected = aug_1 + aug_2 + aug_3 + aug_4    
    aug_T = np.copy(sample)  
    aug_T[selected_indices] = aug_selected  

    return aug_T  


def DataTransform_FD(sample, config):
    """
    Apply weak and strong augmentations in the frequency domain to a subset of signals.

    Args:
    - sample (torch.Tensor or np.ndarray): Shape [features, timesteps], a single sample.
    - config (ConfigAugmentation): Configuration settings.
    - thres (float): Percentage of signals to be perturbed (0-100).

    Returns:
    - torch.Tensor: Augmented sample, shape [features, timesteps].
    """
    sample = torch.tensor(sample, dtype=torch.cfloat) 
    thres = config.thres
    num_features = sample.shape[0] 
    num_perturb = max(1, int(num_features * thres / 100))  
    selected_indices = np.random.choice(num_features, num_perturb, replace=False)

    aug_1 = remove_frequency(sample[selected_indices], pertub_ratio=0.1)
    aug_2 = add_frequency(sample[selected_indices], pertub_ratio=0.1)

    sample[selected_indices] = aug_1 + aug_2  

    return sample 

    
##########################################################################################



def remove_frequency(x, pertub_ratio=0.0):
    """
    Remove certain frequency components by applying a mask.

    Args:
    - x (torch.Tensor): Input sample, shape [features, timesteps].
    - pertub_ratio (float): Percentage of values to be removed.

    Returns:
    - torch.Tensor: Frequency-masked sample.
    """
    mask = torch.rand(x.shape, device=x.device) > pertub_ratio  
    return x * mask

    

def add_frequency(x, pertub_ratio=0.0):
    """
    Add random frequency noise.

    Args:
    - x (torch.Tensor): Input sample, shape [features, timesteps].
    - pertub_ratio (float): Percentage of values to be perturbed.

    Returns:
    - torch.Tensor: Sample with added noise.
    """
    mask = torch.rand(x.shape, device=x.device) > (1 - pertub_ratio) 

    real_max = x.real.amax()
    imag_max = x.imag.amax()
    max_amplitude = torch.max(real_max, imag_max)  
    random_am = torch.rand(x.shape, device=x.device) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am

    return x + pertub_matrix




def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, keepratio=0.9, mask='binomial'):
    global mask_id


    if isinstance(x, np.ndarray):
        nan_mask = ~np.isnan(x).any(axis=-1)  
        x[~nan_mask] = 0
    elif isinstance(x, torch.Tensor):
        nan_mask = ~torch.isnan(x).any(dim=-1)  
        x[~nan_mask] = 0
    else:
        raise TypeError("Unsupported type for masking function: {}".format(type(x)))

    if isinstance(x, np.ndarray):
        if mask == 'binomial':
            mask_id = np.random.binomial(1, keepratio, size=x.shape).astype(bool)
        x[~mask_id] = 0
    else:  
        if mask == 'binomial':
            mask_id = torch.bernoulli(torch.full(x.shape, keepratio, device=x.device)).bool()
        x[~mask_id] = 0

    return x


def jitter(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    """
    Apply scaling augmentation to a single sample.

    Args:
    - x (np.ndarray): Input sample, shape [features, timesteps].
    - sigma (float): Scaling factor variance.

    Returns:
    - np.ndarray: Scaled sample, shape [features, timesteps].
    """
    x = np.array(x)
    if len(x.shape) == 2:  
        features, timesteps = x.shape
        factor = np.random.normal(loc=2., scale=sigma, size=(features, timesteps))
        return x * factor
    else:
        raise ValueError("Scaling function expected shape [features, timesteps], but got: {}".format(x.shape))


def permutation(x, max_segments=5, seg_mode="random"):
    """
    Apply permutation augmentation to a single sample.

    Args:
    - x (np.ndarray): Input sample, shape [features, timesteps].
    - max_segments (int): Maximum number of segments.
    - seg_mode (str): "random" for random segment selection, otherwise uses even splits.

    Returns:
    - np.ndarray: Permuted sample, shape [features, timesteps].
    """
    x = np.array(x) 

    if len(x.shape) != 2: 
        raise ValueError("Permutation function expected shape [features, timesteps], but got: {}".format(x.shape))

    features, timesteps = x.shape
    orig_steps = np.arange(timesteps)

    num_segs = np.random.randint(1, max_segments + 1)  # Ensure at least 1 segment

    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(timesteps - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)    
        permuted_indices = np.concatenate([np.random.permutation(segment) for segment in splits])
        return x[:, permuted_indices] 
    return x  


    
