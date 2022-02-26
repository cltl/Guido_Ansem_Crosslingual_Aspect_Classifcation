import torch
import itertools
from collections import Counter

def get_class_weights(labels):
    counts = Counter(list(itertools.chain(*labels)))
    weights = [sum(counts.values())/(len(counts) * value) for value in counts.values()]
    return torch.tensor(weights)


def init_loss_function(labels, smoothing=0):
    class_weights = get_class_weights(labels)
    class_weights = class_weights ** smoothing
    critereon = torch.nn.CrossEntropyLoss(weight=class_weights)
    return critereon

def calculate_loss(outputs, labels, critereon, device):
    pred = torch.flatten(outputs.logits, end_dim=1).to(device)
    lab = torch.flatten(labels).to(device)
    loss = critereon(pred.to('cpu'), lab.to('cpu'))
    return loss