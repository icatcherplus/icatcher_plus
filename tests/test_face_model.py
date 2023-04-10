import sys

sys.path.append("./reproduce/")

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from PIL import Image
from reproduce.face_classifier.fc_data import ImageFolderWithAge, create_train_test
from reproduce.face_classifier.fc_train import (
    get_args,
    get_loss,
    init_face_classifier,
    make_optimizer_and_scheduler,
    train_face_classifier,
)
from torchvision import transforms


@pytest.fixture
def num_images():
    return 200


@pytest.fixture
def transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def create_mock_dataset(outdir, n=200):
    # Mock raw data for testing
    for im_cls in ["infants", "non_infants"]:
        Path(outdir, im_cls).mkdir()

    gt = {}
    gt["num_images"] = n
    gt["images"] = []
    gt["ages"] = np.concatenate(
        [
            np.random.randint(0, 3, size=(n // 2,)),
            np.random.randint(3, 60, size=(n // 2,)),
        ]
    )
    np.random.shuffle(gt["ages"])
    for idx in range(n):
        age = gt["ages"][idx]
        im_cls = "infants" if age <= 2 else "non_infants"

        imarray = np.random.rand(100, 100, 3) * 255
        fname = Path(outdir, f"{im_cls}/{age}_{idx}.png")
        Image.fromarray(imarray.astype("uint8")).save(fname)

        gt["images"].append(fname)
    return gt


def test_image_folder_with_age_dataset(transform, num_images):
    with TemporaryDirectory() as outdir:
        create_mock_dataset(outdir, num_images)
        dataset = ImageFolderWithAge(root=outdir, transform=transform)
        assert len(dataset) == num_images
        for _, label, age in dataset:
            assert label == (0 if age <= 2 else 1)


def test_dataset_preprocess_split(num_images):
    with TemporaryDirectory() as outdir:
        gt = create_mock_dataset(outdir, num_images)
        create_train_test(outdir, 0.7)

        min_ct = sum([a <= 2 for a in gt["ages"]])
        min_ct = min(min_ct, num_images - min_ct)

        train_counter = sum(
            [file.is_file() for file in Path(outdir, "train").glob("**/*")]
        )
        assert train_counter == 2 * int(0.7 * min_ct)

        val_counter = sum([file.is_file() for file in Path(outdir, "val").glob("**/*")])
        assert val_counter == 2 * int(0.15 * min_ct)

        test_counter = sum(
            [file.is_file() for file in Path(outdir, "test").glob("**/*")]
        )
        assert test_counter == 2 * int(0.15 * min_ct)


@pytest.mark.parametrize(
    "model_name",
    {
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "resnet18",
        "wide_resnet",
    },
)
def test_train_fc_classifiers(model_name, transform):
    args = get_args([])
    args.epochs = 1
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    with TemporaryDirectory() as dataset_dir, TemporaryDirectory() as save_dir:
        dataloaders = {}
        create_mock_dataset(dataset_dir, 32)
        create_train_test(dataset_dir, 0.5)
        for phase in ["train", "val"]:
            dataset = ImageFolderWithAge(
                root=str(Path(dataset_dir, phase)), transform=transform
            )
            dataloaders[phase] = torch.utils.data.DataLoader(dataset)

        args.model = model_name
        model, _ = init_face_classifier(args, model_name)
        criterion = get_loss()
        optimizer, scheduler = make_optimizer_and_scheduler(args, model)
        train_face_classifier(
            args,
            model.to(args.device),
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            save_dir=save_dir,
        )

        assert len(set(Path(save_dir).glob("**/*"))) == 3
        assert (Path(save_dir) / "log.txt").is_file()
        assert (Path(save_dir) / "weights_best.pt").is_file()
        assert (Path(save_dir) / "weights_last.pt").is_file()


def test_train_fc_classifier_reg_age(transform):
    args = get_args([])
    args.epochs = 1
    args.regularize_age = True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    with TemporaryDirectory() as dataset_dir, TemporaryDirectory() as save_dir:
        dataloaders = {}
        create_mock_dataset(dataset_dir, 32)
        create_train_test(dataset_dir, 0.5)
        for phase in ["train", "val"]:
            dataset = ImageFolderWithAge(
                root=str(Path(dataset_dir, phase)), transform=transform
            )
            dataloaders[phase] = torch.utils.data.DataLoader(dataset)

        model, _ = init_face_classifier(args, args.model)
        criterion = get_loss()
        optimizer, scheduler = make_optimizer_and_scheduler(args, model)
        train_face_classifier(
            args,
            model.to(args.device),
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            save_dir=save_dir,
        )

        assert len(set(Path(save_dir).glob("**/*"))) == 3
        assert (Path(save_dir) / "log.txt").is_file()
        assert (Path(save_dir) / "weights_best.pt").is_file()
        assert (Path(save_dir) / "weights_last.pt").is_file()
