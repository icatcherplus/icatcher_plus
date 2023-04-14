from torchvision import transforms, datasets
import torch.utils.data.dataloader as dataloader
from pathlib import Path
from .fc_eval import get_fc_data_transforms
from collections import defaultdict
import numpy as np
import shutil
import os

def create_train_test(raw_data_folder, train_val_split = 0.7):
    infant_folder = Path(raw_data_folder, "infants")
    non_infant_folder = Path(raw_data_folder, "non_infants")
    non_infants = []
    infants = []
    non_infant_participants = defaultdict(list)
    infant_participants = defaultdict(list)
    for file in non_infant_folder.glob("*"):
        participant = file.stem.split("_")[0]
        non_infant_participants[participant].append(file)
    for k in non_infant_participants.keys():
        my_list = non_infant_participants[k]
        max_range = min(len(my_list), 50)
        for i in range(max_range):
            non_infants.append(my_list[i].name)
    for file in infant_folder.glob("*"):
        participant = file.stem.split("_")[0]
        infant_participants[participant].append(file)
    for k in infant_participants.keys():
        my_list = infant_participants[k]
        max_range = min(len(my_list), 50)
        for i in range(max_range):
            infants.append(my_list[i].name)
    print("infants: {}, non_infants: {}".format(len(infants), len(non_infants)))
    dirs = {}
    for split in ["train", "val", "test"]:
        dirs[split] = {}
        for c in ["infant", "non_infant"]:
            split_dir = Path(raw_data_folder, split, c)
            if split_dir.is_dir():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True)
            dirs[split][c] = split_dir
    my_range = min(len(infants), len(non_infants))
    indices = np.arange(my_range)
    train = np.random.choice(indices, size=int(my_range*train_val_split), replace=False)
    non_train = np.setdiff1d(indices, train)
    val = np.random.choice(non_train, size=len(non_train) // 2, replace=False)
    test = np.setdiff1d(non_train, val)
    for index in train:
        shutil.copyfile(Path(infant_folder, infants[index]), Path(dirs["train"]["infant"], infants[index]))
        shutil.copyfile(Path(non_infant_folder, non_infants[index]), Path(dirs["train"]["non_infant"], non_infants[index]))
    for index in val:
        shutil.copyfile(Path(infant_folder, infants[index]), Path(dirs["val"]["infant"], infants[index]))
        shutil.copyfile(Path(non_infant_folder, non_infants[index]), Path(dirs["val"]["non_infant"], non_infants[index]))
    for index in test:
        shutil.copyfile(Path(infant_folder, infants[index]), Path(dirs["test"]["infant"], infants[index]))
        shutil.copyfile(Path(non_infant_folder, non_infants[index]), Path(dirs["test"]["non_infant"], non_infants[index]))

def get_dataset_dataloaders(args, input_size, batch_size, shuffle=True, num_workers=0):
    data_transforms = get_fc_data_transforms(args, input_size)
    face_data_folder = args.dataset_folder
    create_train_test(face_data_folder)
    # Create training and validation datasets
    image_datasets = {
        'train': ImageFolderWithAge(str(Path(face_data_folder, 'train')), data_transforms['train']),
        'val': ImageFolderWithAge(str(Path(face_data_folder, 'val')), data_transforms['val']),
        'test': ImageFolderWithAge(str(Path(face_data_folder, 'test')), data_transforms['test']),
    }
    # print('\n\nImageFolder class to idx: ', image_datasets['val'].class_to_idx)
    # infant - 0, target - 1
    print("# train samples:", len(image_datasets['train']))
    print("# validation samples:", len(image_datasets['val']))
    print("# test samples:", len(image_datasets['test']))

    # Create training and validation dataloaders, never shuffle val and test set
    dataloaders_dict = {x: dataloader.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=False if x != 'train' else shuffle,
                                                 num_workers=num_workers) for x in data_transforms.keys()}
    return dataloaders_dict

class ImageFolderWithAge(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)        
        path = self.imgs[index][0]
        age = os.path.basename(path).split("_")[0]
        if not age.isdigit():
            age = 2
        return (img, label, int(age))
