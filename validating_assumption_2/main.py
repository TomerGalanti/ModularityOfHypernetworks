from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class Net(nn.Module):
    def __init__(self, dataset, width):
        super(Net, self).__init__()

        if dataset == 'MNIST' or dataset == 'FASHION':
            d = 28*28
        else:
            d = 3*32*32
        self.d = d

        self.fc1 = nn.Linear(d, width)
        self.fc2 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1,self.d)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        output = x
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output,
                    F.one_hot(target, num_classes=10).float(), reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Validating Assumption 2')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='FASHION',
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.dataset == 'MNIST':

        dataset1 = datasets.MNIST('./data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('./data', train=False,
                           transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    elif args.dataset == 'CIFAR':

        dataset1 = datasets.CIFAR10(root='./data', transform=transform,
                           train=True, download=True)
        dataset2 = datasets.CIFAR10(root='./data', transform=transform,
                           train=False, download=True)
        train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    elif args.dataset == 'FASHION':

        dataset1 = datasets.FashionMNIST(root='./data', transform=transform,
                           train=True, download=True)
        dataset2 = datasets.FashionMNIST(root='./data', transform=transform,
                           train=False, download=True)
        train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    errs = []
    for i in range(100):

        errors_gt = []
        for width in range(1,30):
            model = Net(args.dataset, width).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                err = test(model, device, test_loader)

            errors_gt += [err]

            print (errors_gt)

        errs += [errors_gt]

        means_gt = (np.array(errs)).mean(axis=0)
        stds_gt = (np.array(errs)).std(axis=0)

        print('GT:')
        print(means_gt.tolist())
        print(stds_gt.tolist())


if __name__ == '__main__':
    main()