import argparse
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
# import visualize
from face_classifier.fc_data import *
from face_classifier.fc_eval import *
from face_classifier.fc_model import *
from scipy.stats import beta
from tqdm import tqdm


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--device', '-d', default='cuda:0', type=str)
    parser.add_argument('--seed', default=0, type=int)
    # augmentations
    parser.add_argument('--rotation', default=False, action='store_true')
    parser.add_argument('--cropping', default=False, action='store_true')
    parser.add_argument('--hor_flip', default=False, action='store_true')
    parser.add_argument('--ver_flip', default=False, action='store_true')
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--erasing', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    # model architecture
    parser.add_argument('--model', default='vgg16', type=str)  # resnet, alexnet, vgg, squeezenet
    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)  # SGD, Adam
    # learning rate
    parser.add_argument('--lr', default=0.001, type=float)
    # batch size
    parser.add_argument('--bs', default=8, type=int)
    # number of epochs, epochs=0 for evalusation only
    parser.add_argument('--epochs', default=40, type=int)
    # schedule
    parser.add_argument('--scheduler', default='none', type=str)  # exp, ms
    # dropout
    parser.add_argument('--dropout', default=0, type=float)  # can only be applied to resnet & densenet
    # comments
    parser.add_argument('--suffix', '-s', default='', type=str)
    # regularize based on estimated age
    parser.add_argument('--regularize_age', default=False, action='store_true', help="regularize loss based on estimated age")
    # resume from a path
    parser.add_argument('--resume', default=None, type=str)
    # generate the test json or not
    parser.add_argument('--test', default=False, action='store_true')
    # eval on training set or not
    parser.add_argument('--eval_train', default=False, action='store_true')
    # only eval on validation set
    parser.add_argument('--eval_val', default=False, action='store_true')
    parser.add_argument('--model_fp', default=None, type=str)
    # plot learning curve or not
    parser.add_argument('--plot_lrcv', default=False, action='store_true')
    parser.add_argument("--dataset_folder", default="", help="path to processed dataset folder (not the raw one)")
    args = parser.parse_args(args)
    return args


def train_face_classifier(args, model, dataloaders, criterion, optimizer, scheduler, save_dir=None,
                          plot_lr_curve=False):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    since = time.time()
    best_acc, best_epoch = 0.0, 0
    train_acc_history, train_loss_history = [], []
    val_acc_history, val_loss_history = [], []

    beta_dist = beta.pdf(x=np.arange(0.005, 1, 0.01), a=1.3, b=10)
    beta_dist = (1 - (beta_dist - beta_dist.min()) / np.max(beta_dist - beta_dist.min())) / 10

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_samples = len(dataloaders[phase].dataset)
            
            # Iterate over data
            for inputs, labels, ages in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                
                loss_weights = 1 + torch.from_numpy(np.asarray(beta_dist[ages])).to(args.device).float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if args.regularize_age:
                        loss = (loss_weights * loss)
                    loss = loss.mean()
                    _, preds = torch.max(outputs, 1)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(torch.eq(preds, labels)).item()
                # num_samples += inputs.size(0)

            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects / num_samples
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            with open((save_dir / 'log.txt'), 'a+') as f:
                f.write(f'Epoch{epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            if phase == 'train' and scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                print('Learning rate:', lr)

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc, best_epoch = epoch_acc, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, (save_dir / 'weights_best.pt'))
                last_model_wts = copy.deepcopy(model.state_dict())
                torch.save(last_model_wts, (save_dir / 'weights_last.pt'))
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

    if plot_lr_curve:
        visualize.plot_learning_curve(train_loss_history, val_loss_history, save_dir=save_dir, isLoss=True)
        visualize.plot_learning_curve(train_acc_history, val_acc_history, save_dir=save_dir, isLoss=False)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    best_acc_str = 'Best val Acc: %.4f, epoch %d\n' % (best_acc, best_epoch)
    with open((save_dir / 'log.txt'), 'a+') as f:
        f.write(best_acc_str)
    print(best_acc_str)

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    args = get_args()
    seed_everything(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU available!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    model_name = args.model

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    # You should use a power of 2.
    batch_size = args.bs

    # Shuffle the input data?
    shuffle_datasets = True

    # Number of epochs to train for
    num_epochs = args.epochs

    ### IO
    # Path to a model file to use to start weights at
    resume_from = None if args.resume == 'none' else args.resume

    # Directory to save weights to
    save_dir = "../models/fc_weights"
    save_dir += "-A"
    if args.rotation:
        save_dir += "r"
    if args.cropping:
        save_dir += "c"
    if args.color:
        save_dir += "o"
    if args.hor_flip:
        save_dir += "h"
    if args.ver_flip:
        save_dir += "v"
    if args.erasing:
        save_dir += "e"
    if args.noise:
        save_dir += "n"
    save_dir += "-M" + model_name
    save_dir += '-D' + str(args.dropout)
    save_dir += "-O" + args.optimizer
    save_dir += '-S' + args.scheduler
    save_dir += "-L" + str(args.lr)
    save_dir += "-B" + str(batch_size)
    save_dir += '-s' + str(args.seed)
    save_dir += args.suffix

    os.makedirs(save_dir, exist_ok=True)

    # Save weights for all epochs, not just the best one
    save_all_epochs = False

    # Initialize the model for this run
    model, input_size = init_face_classifier(args, model_name=model_name, num_classes=num_classes,
                                             resume_from=resume_from)
    dataloaders = get_dataset_dataloaders(args, input_size, batch_size, shuffle_datasets)
    criterion = get_loss()

    # Move the model to the gpu if needed
    model = model.to(args.device)

    optimizer, scheduler = make_optimizer_and_scheduler(args, model)

    # Train the model!
    train_face_classifier(args=args, model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, save_dir=Path(save_dir), plot_lr_curve=args.plot_lrcv)

    model.load_state_dict(torch.load((Path(save_dir) / 'weights_last.pt')))
    generate_train_labels = False
    generate_val_labels = True
    if args.eval_train:
        train_loss, train_top1, train_labels, train_probs, _ = evaluate(args, model, dataloaders['train'], criterion,
                                                                        is_labelled=True,
                                                                        generate_labels=generate_train_labels)
        print(f'train_loss: {train_loss:.4f}', f'train_top1: {train_top1:.4f}')

    val_loss, val_top1, val_labels, val_probs, val_target_labels = evaluate(args, model, dataloaders['val'], criterion,
                                                                            return_prob=False, is_labelled=True,
                                                                            generate_labels=generate_val_labels)

    test_loss, test_top1, test_labels, test_probs, test_target_labels = evaluate(args, model, dataloaders['test'], criterion,
                                                                            return_prob=False, is_labelled=True,
                                                                            generate_labels=generate_val_labels)

    print("\n[val] Failed images:")
    err_idxs = np.where(np.array(val_labels) != np.array(val_target_labels))[0]
    # visualize.print_data_img_name(dataloaders, 'val', err_idxs)
    print()
    print(f'val_loss: {val_loss:.4f}', f'val_top1: {val_top1:.4f}')
    print(f'test_loss: {test_loss:.4f}', f'test_top1: {test_top1:.4f}')
    with open((Path(save_dir) / 'log.txt'), "a+") as f:
        f.write(
            f'val_loss: {val_loss:.4f}', f'val_top1: {val_top1:.4f}\n' +
            f'test_loss: {test_loss:.4f}', f'test_top1: {test_top1:.4f}'
        )
