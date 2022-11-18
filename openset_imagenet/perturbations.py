"""Helper functions to create perturbations"""
import torch
import vast


def fgsm_attack(clean_im, epsilon, grad, negs_label, device):
    """ Generates adversarial samples and the corresponding labels. Pixel values are clipped between
    [0,1]. Parts taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html.

    Args:
        clean_im: Tensor of clean images.
        epsilon: Per-pixel attack magnitude.
        grad: Loss gradient with respect to clean images.
        negs_label: Label of generated samples
        device: Current cuda device.

    Returns:
         perturbed_im: Tensor containing perturbed samples.
         labels: Tensor containing perturbed samples labels
    """
    sign_data_grad = grad.sign()
    perturbed_im = clean_im + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_im = torch.clamp(perturbed_im, min=0.0, max=1.0)
    labels = torch.ones(clean_im.shape[0], device=device) * negs_label
    return perturbed_im, labels


def decay_epsilon(start_eps, mu, curr_epoch, wait_epochs, lower_bound=0.01):
    """ Decays an initial epsilon [start_eps], waiting a number of epochs [wait_epochs], using a
    base factor [mu]. Pixel values are clipped in [0,1].

    Args:
        start_eps: Initial epsilon value.
        mu: Base of the decaying function.
        curr_epoch: Current training epoch.
        wait_epochs: Number of epochs to wait for every decay.
        lower_bound: Minimum epsilon to return.

    Returns:
        New epsilon value.
    """
    return max(start_eps * mu**(curr_epoch//wait_epochs), lower_bound)


class Noise:
    """Simple class that groups noise distributions"""

    def __init__(self, noise_type, **kwargs):
        """Noise distributions require different set of parameters as follows:
            uniform: low,high
            gaussian: loc, std
            bernoulli: prob, epsilon
        """
        self.noise_type = noise_type
        self.epsilon = None

        def cuda_by_default(value):
            """Used to convert the class parameters into float cuda tensors, this is to create the
            sampled noise directly on the cuda device, avoiding to move the sampled noise from
            cpu to gpu"""
            return torch.tensor(value, dtype=torch.float, device=vast.tools._device)

        if noise_type == "gaussian":
            self.dist = torch.distributions.Normal(loc=cuda_by_default(kwargs.get("loc")),
                                                   scale=cuda_by_default(kwargs.get("std")))
        elif noise_type == "uniform":
            self.dist = torch.distributions.Uniform(low=cuda_by_default(kwargs.get("low")),
                                                    high=cuda_by_default(kwargs.get("high")))
        elif noise_type == "bernoulli":
            self.dist = torch.distributions.Bernoulli(probs=cuda_by_default(kwargs.get("prob")))
            self.sign = torch.distributions.Uniform(low=cuda_by_default(-1),
                                                    high=cuda_by_default(1))
            self.epsilon = kwargs.get('epsilon')
        else:
            raise Exception("Select one known method: gaussian, uniform, bernoulli")

    def perturb(self, clean_im):
        """Returns a perturbed image with noise"""
        if self.noise_type == 'bernoulli':
            noise = (torch.sign(self.sign.sample(clean_im.shape))
                     * self.epsilon
                     * self.dist.sample(clean_im.shape))
        else:
            noise = self.dist.sample(clean_im.shape)
        return torch.clamp(clean_im + noise, min=0.0, max=1.0)

    @staticmethod
    def get_labels(shape, device, negs_label=-1):
        """Return labels of negative samples"""
        return torch.ones(shape, device=device) * negs_label
