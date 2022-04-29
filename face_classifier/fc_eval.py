import os
from torch.autograd import Variable
# from torchvision import datasets, transforms
# from config import multi_face_folder
from data import *


def get_fc_data_transforms(args, input_size, dt_key=None):
    if dt_key is not None and dt_key != 'train':
        return {dt_key: transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.CenterCrop(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    # Apply data augmentation
    aug_list = []
    aug_list.append(transforms.ToTensor())
    if args.cropping:
        aug_list.append(transforms.RandomResizedCrop((input_size, input_size)))
    else:
        aug_list.append(transforms.Resize((input_size, input_size), antialias=True))
    if args.rotation:
        aug_list.append(transforms.RandomRotation(20))
    if args.color:
        aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if args.hor_flip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if args.ver_flip:
        aug_list.append(transforms.RandomVerticalFlip())
    aug_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    if args.noise:
        aug_list.append(AddGaussianNoise(0, 0.1))
    if args.erasing:
        aug_list.append(transforms.RandomErasing())

    aug_transform = transforms.Compose(aug_list)

    # Define data transformation on train, val, test set respectively
    data_transforms = {
        'train': aug_transform,
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.CenterCrop(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def evaluate(args, model, dataloader, criterion, return_prob=False, is_labelled=False, generate_labels=True):
    model.eval()
    running_loss = 0
    running_top1_correct = 0
    pred_labels, pred_probs = [], []
    target_labels = []

    # Iterate over data.
    for inputs, labels in dataloader:
        if generate_labels:
            target_labels.extend(list(labels.numpy()))  # -- 1d array

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        # track history if only in train
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            if is_labelled:
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)  # Make prediction

            if return_prob:
                pred_probs.append(outputs.cpu().detach().numpy())
            if generate_labels:
                nparr = preds.cpu().detach().numpy()  # -- 1d array
                pred_labels.extend(nparr)

        if is_labelled:
            running_loss += loss.item() * inputs.size(0)
            running_top1_correct += torch.sum(preds == labels.data)
        else:
            pass

    if is_labelled:
        epoch_loss = float(running_loss / len(dataloader.dataset))
        epoch_top1_acc = float(running_top1_correct.double() / len(dataloader.dataset))
    else:
        epoch_loss = None
        epoch_top1_acc = None

    if return_prob:
        print("Predicted label softmax output:", pred_probs)

    # Show confusion matrix
    # if generate_labels and is_labelled:
        # print("pred labels:", np.shape(pred_labels), pred_labels)
        # print("target labels:", np.shape(target_labels), target_labels)
        # cm = confusion_mat(target_labels, pred_labels, classes=['infant', 'others'])
        # print("Confusion matrix:\n", cm)

    return epoch_loss, epoch_top1_acc, pred_labels, pred_probs, target_labels


# TODO: This is untested! Predict on a smaller subset of images than batch size
def predict_on_minibatch(args, inputsize, test_imgs, model):
    model.eval()
    pred_labels, pred_probs = [], []

    # Apply test data transform
    transform = get_fc_data_transforms(args, inputsize, 'test')['test']

    for img in test_imgs:
        test_img = Variable(transform(img).float(), requires_grad=True)
        test_img = test_img.unsqueeze(0)  # for vgg, may not be needed for resnet. TODO: double check
        test_img = test_img.to(args.device)
        test_out = model(test_img)
        _, preds = torch.max(test_out, 1)
        pred_labels.append(preds)  # unpack?
        pred_probs.append(test_out)  # unpack?

    return pred_labels, pred_probs


def predict_on_test(args, model, dataloaders, criterion):
    # Get predictions for the test set
    _, _, test_labels, test_probs, _ = evaluate(args, model, dataloaders['test'], criterion,
                                                return_prob=False, is_labelled=False, generate_labels=True)

    ''' These convert your dataset labels into nice human readable names '''

    def label_number_to_name(lbl_ix):
        return dataloaders['val'].dataset.classes[lbl_ix]

    # TODO: modify this
    def dataset_labels_to_names(dataset_labels, dataset_name):
        # dataset_name is one of 'train','test','val'
        # replace with multi-face-folder loc (usually under preprocessed dataset path / "multi_face")
        dataset_root = os.path.join(multi_face_folder, dataset_name)
        found_files = []
        for parentdir, subdirs, subfns in os.walk(dataset_root):
            parentdir_nice = os.path.relpath(parentdir, dataset_root)
            found_files.extend([os.path.join(parentdir_nice, fn) for fn in subfns if fn.endswith('.png')])
        # Sort alphabetically, this is the order that our dataset will be in
        found_files.sort()
        # Now we have two parallel arrays, one with names, and the other with predictions
        assert len(found_files) == len(dataset_labels), "Found more files than we have labels"
        preds = {os.path.basename(found_files[i]): list(map(label_number_to_name, dataset_labels[i])) for i in
                 range(len(found_files))}
        return preds

    output_test_labels = "test_set_predictions"
    output_salt_number = 0

    output_label_dir = "."

    while os.path.exists(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))):
        output_salt_number += 1
        # Find a filename that doesn't exist

    # test_labels_js = dataset_labels_to_names(test_labels, "test")
    # with open(os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number)), "w") as f:
    #   json.dump(test_labels_js, f, sort_keys=True, indent=4)

    print("Wrote predictions to:\n%s" % os.path.abspath(
        os.path.join(output_label_dir, '%s%d.json' % (output_test_labels, output_salt_number))))

    return
