import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

class Uniform(Dataset):
    def __init__(self, **kwargs):
        super(Uniform, self).__init__(**kwargs)

    def __len__(self):
        return 10000

    def __getitem__(self, index):

        return torch.FloatTensor(1, 28, 28).uniform_(-1, 1), 1


def get_dims(args):
    if args.dataset == 'MNIST':
        h = 28
        nc = 1
    elif args.dataset == 'uniform':
        h = 28
        nc = 1
    elif args.dataset == 'CIFAR':
        h = 32
        nc = 3
    return nc, h


def get_data(args, kwargs):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.dataset == 'MNIST':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', transform=transform,
                           train=True, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', transform=transform,
                           train=False, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'uniform':

        uniform_dataset = Uniform()

        train_loader = torch.utils.data.DataLoader(
            uniform_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            uniform_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'CIFAR':

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', transform=transform,
                             train=True, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', transform=transform,
                             train=False, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

