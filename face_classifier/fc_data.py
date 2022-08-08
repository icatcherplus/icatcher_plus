from torchvision import transforms, datasets
import torch.utils.data.dataloader as dataloader
from pathlib import Path
from .fc_eval import get_fc_data_transforms
from collections import defaultdict
import numpy as np
import shutil


def create_train_test(raw_data_folder):
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
        print(len(my_list))
        max_range = min(len(my_list), 50)
        for i in range(max_range):
            non_infants.append(my_list[i].name)
    for file in infant_folder.glob("*"):
        participant = file.stem.split("_")[0]
        infant_participants[participant].append(file)
    for k in infant_participants.keys():
        my_list = infant_participants[k]
        print(len(my_list))
        max_range = min(len(my_list), 50)
        for i in range(max_range):
            infants.append(my_list[i].name)
    print("infants: {}, non_infants: {}".format(len(infants), len(non_infants)))
    train_dir_infant = Path(raw_data_folder, "train", "infant")
    train_dir_non_infant = Path(raw_data_folder, "train", "non_infant")
    val_dir_infant = Path(raw_data_folder, "val", "infant")
    val_dir_non_infant = Path(raw_data_folder, "val", "non_infant")
    train_dir_infant.mkdir(parents=True, exist_ok=True)
    train_dir_non_infant.mkdir(parents=True, exist_ok=True)
    val_dir_infant.mkdir(parents=True, exist_ok=True)
    val_dir_non_infant.mkdir(parents=True, exist_ok=True)
    my_range = min(len(infants), len(non_infants))
    indices = np.arange(my_range)
    train_val_split = 0.8
    train = np.random.choice(indices, size=int(my_range*train_val_split), replace=False)
    val = np.setdiff1d(indices, train)
    for index in train:
        shutil.copyfile(Path(infant_folder, infants[index]), Path(train_dir_infant, infants[index]))
        shutil.copyfile(Path(non_infant_folder, non_infants[index]), Path(train_dir_non_infant, non_infants[index]))
    for index in val:
        shutil.copyfile(Path(infant_folder, infants[index]), Path(val_dir_infant, infants[index]))
        shutil.copyfile(Path(non_infant_folder, non_infants[index]), Path(val_dir_non_infant, non_infants[index]))
    counter = 0
    for file in Path(raw_data_folder, "train").glob("**/*"):
        if file.is_file():
            counter += 1
    assert counter == len(train) * 2
    counter = 0
    for file in Path(raw_data_folder, "val").glob("**/*"):
        if file.is_file():
            counter += 1
    assert counter == len(val) * 2


def get_dataset_dataloaders(args, input_size, batch_size, shuffle=True, num_workers=0):
    data_transforms = get_fc_data_transforms(args, input_size)
    face_data_folder = args.dataset_folder
    create_train_test(face_data_folder)
    # Create training and validation datasets
    image_datasets = {'train': datasets.ImageFolder(str(Path(face_data_folder, 'train')), data_transforms['train']),
                      'val': datasets.ImageFolder(str(Path(face_data_folder, 'val')), data_transforms['val']),
                      }
    # print('\n\nImageFolder class to idx: ', image_datasets['val'].class_to_idx)
    # infant - 0, target - 1
    print("# train samples:", len(image_datasets['train']))
    print("# validation samples:", len(image_datasets['val']))

    # Create training and validation dataloaders, never shuffle val and test set
    dataloaders_dict = {x: dataloader.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=False if x != 'train' else shuffle,
                                                 num_workers=num_workers) for x in data_transforms.keys()}
    return dataloaders_dict
