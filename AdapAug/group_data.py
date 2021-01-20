import numpy as np
import torchvision

from PIL import Image
from torch.utils.data import Dataset

class GroupDataset(Dataset): 
    def __init__(self, dataname, split='test', download=False, transform=None, target_transform=None, **kwargs):
        dataset = []
        assert len(dataname) > 1
        for _name in dataname:
            if 'CIFAR' in _name:
                cifar_dataset = torchvision.datasets.__dict__[_name](transform=None, train=('train' in split), **kwargs)
                cifar_dataset.n_cls = 10
                cifar_dataset.labels = np.array(cifar_dataset.targets)
                dataset.append(cifar_dataset)
            elif 'SVHN' in _name:
                svhn_dataset = torchvision.datasets.__dict__[_name](transform=None, split=split, **kwargs)
                svhn_dataset.n_cls = 10
                svhn_dataset.data = np.transpose(svhn_dataset.data, (0,2,3,1))
                dataset.append(svhn_dataset)
                
        self.data = np.concatenate([data.data for data in dataset], axis=0)
        n_cls_list = [data.n_cls for data in dataset]
        accumulated_n_cls_list = [sum(n_cls_list[:i]) for i in range(len(n_cls_list))]
        self.targets = self.labels = sum([list(data.labels + accumulated_n_cls_list[i]) for i, data in enumerate(dataset)],[])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index): 
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target