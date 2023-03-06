import torch
import copy
from pathlib import Path
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
import logging


class MyModel:
    """
    generic container class for network(torch Module), optimizer and scheduler.
    """
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.loss_fn = self.get_loss_fn()
        self.network = self.get_network()
        if self.opt.continue_train:
            self.load_network("latest")
        if self.opt.distributed:
            self.network = DDP(self.network, device_ids=[self.opt.rank])
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def get_loss_fn(self):
        if self.opt.loss == "cat_cross_entropy":
            fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        return fn

    def get_network(self):
        """
        picks a network according to architecture
        :return: the network
        """
        if self.opt.architecture == "fc":
            network = FullyConnected(self.opt)
        elif self.opt.architecture == "icatcher+":
            network = GazeCodingModel(self.opt)
        elif self.opt.architecture == "rnn":
            network = RNNModel(self.opt)
        elif self.opt.architecture == "icatcher_vanilla":
            network = iCatcherOriginal(self.opt)
        else:
            raise NotImplementedError
        network.to(self.opt.device)
        return network

    def get_optimizer(self):
        if self.opt.optimizer == "adam":
            optimizer = torch.optim.Adam(self.network.parameters(),
                                         lr=self.opt.lr,
                                         betas=(0.9, 0.999),
                                         weight_decay=1e-5)
        elif self.opt.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.network.parameters(),
                                        lr=1e-2,
                                        momentum=0.9,
                                        weight_decay=1e-4)
        else:
            raise NotImplementedError
        return optimizer

    def get_scheduler(self):
        """
        picks a scheduler according to lr policy
        :return: the scheduler
        """
        if self.opt.lr_policy == 'lambda':
            lambda_rule = lambda epoch: self.opt.lr_decay_rate ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_policy == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=3)
        elif self.opt.lr_policy == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 5], gamma=0.1)
        elif self.opt.lr_policy == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
                                                          base_lr=self.opt.lr,
                                                          max_lr=self.opt.lr / 20,
                                                          step_size_up=3,
                                                          cycle_momentum=False,
                                                          verbose=True)
        else:
            raise NotImplementedError
        return scheduler

    def load_network(self, which_epoch):
        """
        load model from disk
        :param which_epoch: "latest" = uses the latest version. any other number = some particular epoch.
        :return:
        """
        save_filename = '{}_net.pth'.format(str(which_epoch))
        load_path = Path.joinpath(self.opt.experiment_path, save_filename)
        net = self.network
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        logging.info('loading the model from {}'.format(str(load_path)))
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        try:
            net.load_state_dict(state_dict)
        except RuntimeError:  # deal with old models that were encapsulated with "net"
            new_dict = OrderedDict()
            for i in range(len(state_dict)):
                k, v = state_dict.popitem(False)
                new_k = '.'.join(k.split(".")[1:])
                new_dict[new_k] = v
            net.load_state_dict(new_dict)

    def save_network(self, which_epoch):
        """
        save model to disk.
        :param which_epoch: "latest" = uses the latest version. any other number = some particular epoch.
        :return:
        """
        save_filename = '{}_net.pth'.format(str(which_epoch))
        save_path = Path.joinpath(self.opt.experiment_path, save_filename)
        torch.save(self.network.state_dict(), str(save_path))

    def count_parameters(self):
        """
        retrieve number of parameters in the network
        :return: number of parameters in the network
        """
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


class FullyConnected(torch.nn.Module):
    """
    A straight forward fully-connected neural network, input and output sizes are defined by args.
    """
    def __init__(self, args):
        self.network = torch.nn.ModuleList([
            torch.nn.Flatten(),
            torch.nn.Linear((args.image_size**2)*3*args.sliding_window_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, args.number_of_classes),
        ])

    def forward(self, x):
        for i, layer in enumerate(self):
            x = layer(x)
        return x


class iCatcherOriginal(torch.nn.Module):
    """
    the vanilla iCatcher architecture
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.network = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 16, stride=(1, 1), kernel_size=(3, 3), padding=(0, 0)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(16, 32, stride=1, kernel_size=(3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, stride=1, kernel_size=(3, 3), padding=0),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(32, 64, stride=1, kernel_size=(3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, stride=1, kernel_size=(3, 3), padding=0),
            torch.nn.Flatten(1)
        ])
        self.predictor = Predictor_vanilla().to(self.args.device)
        self.network.to(self.args.device)

    def forward(self, x):
        x = x['imgs']
        embedding = x.view(-1, 3, 100, 100)
        for i, layer in enumerate(self.network):
            embedding = layer(embedding)
        pred = self.predictor(embedding.view(x.shape[0], -1))
        # seq = []
        # out = x
        # for tt in range(x.shape[1]):
        #     out = x[:, tt, :, :, :]
        #     for i, layer in enumerate(self.network):
        #         out = layer(out)
        #     seq.append(out)
        # seq = torch.stack(seq, dim=0)
        # seq = seq.transpose(1, 0)[:, 2, :]
        # return seq
        return pred


class RNNModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pretrained_model = resnet18(pretrained=True)
        modules = list(pretrained_model.children())[:-1]      # delete the last fc layer.
        self.baseModel = torch.nn.Sequential(*modules)
        self.fc1 = torch.nn.Linear(512, 256).to(self.args.device)
        self.bn1 = torch.nn.BatchNorm1d(256).to(self.args.device)
        self.dropout = torch.nn.Dropout(0.2).to(self.args.device)
        self.fc2 = torch.nn.Linear(256, 128).to(self.args.device)
        self.rnn = torch.nn.LSTM(128, 32).to(self.args.device)
        self.fc3 = torch.nn.Linear(32, 3).to(self.args.device)

    def forward(self, x):
        x = x["imgs"]
        b, t, c, h, w = x.shape
        seq = []
        for tt in range(t):
            with torch.no_grad():
                out = self.baseModel(x[:, tt, :, :, :])  # use base model to predict per time-slice, but don't train it.
                out = out.view(out.size(0), -1)  # flatten output
            out = self.fc1(out)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            seq.append(out)
        seq = torch.stack(seq, dim=0)
        out, (h_n, h_c) = self.rnn(seq)
        out = self.fc3(out.transpose(1, 0)[:, 2, :])  # choose RNN_out at the mid time step
        return out


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


class GazeCodingModel3D(torch.nn.Module):
    def __init__(self, device, n, add_box):
        super().__init__()
        self.n = n
        self.device = device
        self.add_box = add_box
        self.encoder_img = Encoder_img_3d()
        self.encoder_box = Encoder_box_seq(n)
        self.predictor = Predictor_fc(2, add_box)

    def forward(self, data):
        imgs = data['imgs'].to(self.device)  # bs x n x 3 x 100 x 100
        boxs = data['boxs'].to(self.device)  # bs x n x 5
        embedding = self.encoder_img(imgs)
        if self.add_box:
            box_embedding = self.encoder_box(boxs.view(boxs.size(0), -1))
            embedding = torch.cat([embedding, box_embedding], -1)
        pred = self.predictor(embedding)
        return pred


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


class Predictor_vanilla(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 64 * 18 * 18 * 5
        self.fc1 = torch.nn.Linear(in_channel, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


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



class Encoder_img_3d(torch.nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.conv1_1 = torch.nn.Conv3d(in_channel, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv1_2 = torch.nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv2_1 = torch.nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.conv2_2 = torch.nn.Conv3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv3_1 = torch.nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.conv3_2 = torch.nn.Conv3d(128, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv4_1 = torch.nn.Conv3d(128, 256, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2))
        self.conv4_2 = torch.nn.Conv3d(256, 256, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv5_1 = torch.nn.Conv3d(256, 512, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv5_2 = torch.nn.Conv3d(512, 512, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn1_1 = torch.nn.BatchNorm3d(16)
        self.bn1_2 = torch.nn.BatchNorm3d(32)
        self.bn2_1 = torch.nn.BatchNorm3d(64)
        self.bn2_2 = torch.nn.BatchNorm3d(64)
        self.bn3_1 = torch.nn.BatchNorm3d(128)
        self.bn3_2 = torch.nn.BatchNorm3d(128)
        self.bn4_1 = torch.nn.BatchNorm3d(256)
        self.bn4_2 = torch.nn.BatchNorm3d(256)
        self.bn5_1 = torch.nn.BatchNorm3d(512)
        self.bn5_2 = torch.nn.BatchNorm3d(512)
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x))) + x
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x))) + x
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x))) + x
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = self.pool(x).view(x.size(0), x.size(1))

        return x


class Encoder_box_seq(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = torch.nn.Linear(5*n, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.bn = torch.nn.BatchNorm1d(512)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(self.bn(x)))
        x = self.fc2(x)

        return x
