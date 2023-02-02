import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models import vgg16
from torchvision import transforms


def get_fc_data_transforms(input_size, dt_key=None):
    if dt_key is not None and dt_key != 'train':
        return {dt_key: transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.CenterCrop(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    # Apply data augmentation
    aug_list = []
    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Resize((input_size, input_size), antialias=True))
    aug_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
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


def init_face_classifier(device, num_classes=2, resume_from=None):
    input_size = 100
    model = vgg16(num_classes=num_classes)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    if resume_from is not None:
        print("Loading weights from %s" % resume_from)
        model.load_state_dict(torch.load(resume_from, map_location=device))
    return model, input_size



class DataTransforms:
    def __init__(self, img_size, mean, std):
        self.transformations = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                # transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

class Predictor_fc(torch.nn.Module):
    def __init__(self, n, add_box):
        super().__init__()
        in_channel = 512 * n if add_box else 256 * n
        self.fc1 = torch.nn.Linear(in_channel, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 3)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        x = self.dropout1(F.relu(self.bn1(x)))
        x = self.fc2(x)
        x = self.dropout2(F.relu(self.bn2(x)))
        x = self.fc3(x)
        return x

class Encoder_box(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(self.bn(x)))
        x = self.fc2(x)

        return x

class GazeCodingModel(torch.nn.Module):
    def __init__(self, args, add_box=True):
        super().__init__()
        self.args = args
        self.n = (args.sliding_window_size + 1) // args.window_stride
        self.add_box = add_box
        self.encoder_img = resnet18(num_classes=256).to(self.args.device)
        self.encoder_box = Encoder_box().to(self.args.device)
        self.predictor = Predictor_fc(self.n, add_box).to(self.args.device)

    def forward(self, data):
        imgs = data['imgs']  # bs x n x 3 x 100 x 100
        boxs = data['boxs']  # bs x n x 5
        embedding = self.encoder_img(imgs.view(-1, 3, 100, 100)).view(-1, self.n, 256)
        if self.add_box:
            box_embedding = self.encoder_box(boxs.view(-1, 5)).view(-1, self.n, 256)
            embedding = torch.cat([embedding, box_embedding], -1)
        pred = self.predictor(embedding)
        return pred