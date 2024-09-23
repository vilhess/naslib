import torch
import numpy as np
import torch.nn as nn 
from torch._functorch.make_functional import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jacfwd, functional_call, vjp, jvp
import random

from . import measure

def subset_classes(inputs, targets, samples_per_class=10, device="cpu", subsample=12):
    dataset_classes = {}
    count_per_class = {}
    class_permutation = None

    for inp, tar in zip(inputs, targets):
        tar = tar.item()
        try:
            if tar not in dataset_classes:
                dataset_classes[tar] = []
                count_per_class[tar] = 0
            if count_per_class[tar] < samples_per_class:
                dataset_classes[tar].append(inp.to(device))
                count_per_class[tar] += 1
        except    Exception as e:
            print(f"Error with target {tar} : {e}")

        if all(count >= samples_per_class for count in count_per_class.values()):
            break

    if len(dataset_classes) > subsample:
        selected_classes = random.sample(list(dataset_classes.keys()), subsample)
        dataset_classes = {key: dataset_classes[key] for key in selected_classes}
        class_permutation = {selected_classes[i]: i for i in range(len(selected_classes))}

    for key in dataset_classes.keys():
        dataset_classes[key] = torch.stack(dataset_classes[key])

    return dataset_classes, class_permutation

class Scalar_NN(nn.Module):
    def __init__(self, network, class_val):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val

    def forward(self, x):
        return self.network(x)[:, self.class_val].reshape(-1, 1)


def model_min_eigenvalue_class(model, x, class_val):

    model = Scalar_NN(network=model, class_val=class_val)

    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0), )).squeeze(0)      

    parameters = {k: v.detach() for k, v in model.named_parameters()}

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    jac1 = vmap(jacrev(fnet_single), (None, 0))(parameters, x)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) if len(j.shape)>2 else j.unsqueeze(2) for j in jac1]

    jac2 = jac1
    operation = 'Naf,Mbf->NMab'
    result = torch.stack([torch.einsum(operation, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    result=result.squeeze()
    u, sigma, v = torch.linalg.svd(result)

    return torch.min(sigma)



@measure("monet", bn=True)
def compute_monet(net, inputs, targets, split_data=1, loss_fn=None):
    dataset_classes, class_permutation = subset_classes(inputs, targets)
    lambdas = []
    for c in dataset_classes.keys():
        x_ntks = dataset_classes[c]
        if class_permutation is not None:
            c = class_permutation[c]
        if len(x_ntks)>1:
            lam = model_min_eigenvalue_class(net, x_ntks, c)
            lambdas.append(lam.cpu().numpy())
    return np.log(np.sum(lambdas))