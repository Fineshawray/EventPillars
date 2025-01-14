import numpy as np
from os import listdir
from os.path import join
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_max, scatter_min
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode
from psee_loader import PSEELoader


class NCaltech101:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)
        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.trans = nn.ModuleList([
            transforms.transforms.RandomHorizontalFlip(0.6),
            transforms.RandomAffine(30, (0.3, 0.3), (0.8, 1.0)),
        ])
        self.traintransforms = transforms.Compose([
            transforms.transforms.RandomApply(self.trans, p=0.5),
            RandomZoom(p=0.5, scale_range=[0.8, 1.2]),
        ])
        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        neg = torch.from_numpy(events[events[:, 3] == -1])
        pos = torch.from_numpy(events[events[:, 3] == 1])
        height = int(max(events[:, 1])) + 1
        width = int(max(events[:, 0])) + 1

        Histogram_neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)
        Histogram_pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)

        t_sequence = events[:, 2] - events[:, 2][0]
        t_sequence = torch.tensor(t_sequence)
        t_values = (t_sequence * 255).clamp(0, 255).byte()
        index = torch.tensor(events[:, 0] + events[:, 1] * width).to(torch.int64)

        # NET
        NET_out, _ = scatter_max(t_values, index, dim=-1, dim_size=height * width)
        NET = NET_out.reshape(height, width)

        # EET
        EET_out, _ = scatter_min(t_values, index, dim=-1, dim_size=height * width)
        EET = EET_out.reshape(height, width)

        tanh = nn.Tanh()

        # AEP
        AEP = tanh(Histogram_neg - Histogram_pos)
        num_events = Histogram_pos + Histogram_neg\

        # ED
        ED = num_events / num_events.max()
        result = torch.stack([NET, EET, AEP, ED], dim=2)

        encoded_tensor= result.float()

        if self.augmentation:
            encoded_tensor = self.traintransforms(encoded_tensor)

        return encoded_tensor, label


class NCars:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)
        self.files = []
        self.labels = []
        self.augmentation = augmentation
        self.trans = nn.ModuleList([
            transforms.transforms.RandomHorizontalFlip(0.6),
            transforms.RandomAffine(30, (0.3, 0.3), (0.8, 1.0)),
        ])
        self.traintransforms = transforms.Compose([
            transforms.transforms.RandomApply(self.trans, p=0.5),
            RandomZoom(p=0.5, scale_range=[0.8, 1.2]),
        ])
        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        f = self.files[idx]
        # Load all events of each sample.
        cur_events = PSEELoader(f).load_delta_t(100000)
        x, y, t, p = cur_events['x'], cur_events['y'], cur_events['t'], cur_events['p']
        events = np.column_stack((x, y, t, p))
        neg = torch.from_numpy(events[events[:, 3] == -1])
        pos = torch.from_numpy(events[events[:, 3] == 1])
        height = int(max(events[:, 1])) + 1
        width = int(max(events[:, 0])) + 1

        Histogram_neg = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)
        Histogram_pos = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * width,
                                       minlength=height * width).reshape(height, width)

        t_sequence = events[:, 2] - events[:, 2][0]
        t_sequence = torch.tensor(t_sequence)
        t_values = (t_sequence * 255).clamp(0, 255).byte()
        index = torch.tensor(events[:, 0] + events[:, 1] * width).to(torch.int64)

        # NET
        NET_out, _ = scatter_max(t_values, index, dim=-1, dim_size=height * width)
        NET = NET_out.reshape(height, width)

        # EET
        EET_out, _ = scatter_min(t_values, index, dim=-1, dim_size=height * width)
        EET = EET_out.reshape(height, width)

        tanh = nn.Tanh()

        # AEP
        AEP = tanh(Histogram_neg - Histogram_pos)
        num_events = Histogram_pos + Histogram_neg\

        # ED
        ED = num_events / num_events.max()
        result = torch.stack([NET, EET, AEP, ED], dim=2)

        encoded_tensor= result.float()

        if self.augmentation:
            encoded_tensor = self.traintransforms(encoded_tensor)

        return encoded_tensor, label


class RandomZoom(torch.nn.Module):
    def __init__(self, p=0.5, scale_range=None, hw=(128, 128)):
        super().__init__()
        if scale_range is None:
            scale_range = [0.80, 1.2]
        self.p = p
        self.scale_range = scale_range
        self.hw = hw

    def forward(self, img):
        if torch.rand(1) < self.p:
            zoom_rand = torch.rand(1) / 2
            zoom_sacle = self.scale_range[0] + zoom_rand * (self.scale_range[1] - self.scale_range[0])
            n_hw = (int(self.hw[0] * zoom_sacle), int(self.hw[1] * zoom_sacle))
            img = F.resize(img, n_hw, InterpolationMode.NEAREST)

            if zoom_rand >= 0.5:
                img = F.center_crop(img, self.hw)
            else:
                pad = (self.hw[0] - n_hw[0]) // 2
                img = F.pad(img, pad)
                img = F.resize(img, self.hw, InterpolationMode.NEAREST)
            return img
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
