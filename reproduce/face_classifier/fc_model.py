import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, models, transforms


def init_face_classifier(args, model_name, num_classes=2, resume_from=None):
    input_size = 100
    model = None

    if model_name == 'vgg11':
        model = models.vgg11(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg13':
        model = models.vgg13(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg19':
        model = models.vgg19(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(num_classes=num_classes)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(num_classes=num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if args.dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(num_ftrs, num_classes))

    elif model_name == "resnet34":
        """ Resnet34
        """
        model = models.resnet34(num_classes=num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if args.dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(num_ftrs, num_classes))

    elif model_name == "resnet50":
        """ Resnet50
        """
        model = models.resnet50(num_classes=num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if args.dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(num_ftrs, num_classes))

    elif model_name == "wide_resnet":
        model = models.wide_resnet50_2(num_classes=num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if args.dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(num_ftrs, num_classes))

    else:
        raise Exception("Invalid model name!")

    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        model.load_state_dict(torch.load(resume_from, map_location=args.device))

    return model, input_size


def make_optimizer_and_scheduler(args, model):
    # Get all the parameters
    params_to_update = model.parameters()
    print(model)

    # Optimizer
    arg_optim = args.optimizer
    if arg_optim == 'adam':
        optimizer = optim.Adam(params_to_update, lr=args.lr)
    elif arg_optim == 'amsgrad':
        optimizer = optim.Adam(params_to_update, lr=args.lr, amsgrad=True)
    elif arg_optim == 'adagrad':
        optimizer = optim.Adagrad(params_to_update, lr=args.lr)
    elif arg_optim == 'sgdo':
        optimizer = optim.SGD(params_to_update, lr=args.lr)
    elif arg_optim == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    elif arg_optim == 'adamwd':
        optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=1e-4)
    elif arg_optim == 'sgdwd':
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise Exception("Invalid optimizer!")

    # Scheduler
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    elif args.scheduler == 'ms':
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20], gamma=0.1)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise Exception("Invalid scheduler!")

    return optimizer, scheduler


def get_loss():
    # Create an instance of the loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    return criterion

# class FaceClassifierModel(nn.Module):
#
#   def __init__(self, device, n, model_name='vgg16', num_classes=2):
#     super().__init__()
#     self.device = device
#     self.n = n
#     model = models.vgg16(num_classes=num_classes)
#     num_ftrs = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(num_ftrs, num_classes)
#     self.model = model
#     # input size = 100, 100
#
#   def forward(self, data):
#
#     pass
