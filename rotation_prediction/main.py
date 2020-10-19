from __future__ import print_function
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rotation_prediction.models import Hypernet, Embedding
from rotation_prediction.datasets import MNIST_ROTATE, CIFAR_ROTATE


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    MSEloss = nn.MSELoss()

    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device).float()
        optimizer.zero_grad()

        output = model(data1, data2)

        if args.task == 'pixels':
            loss = MSEloss(output, target)
        elif args.task == 'rotations':
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target.long())

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    MSEloss = nn.MSELoss()

    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device).float()
            output = model(data1, data2)

            if args.task == 'pixels':
                test_loss += MSEloss(output, target).item() * len(data1) # sum up batch loss
            elif args.task == 'rotations':
                output = F.log_softmax(output, dim=1)
                test_loss += F.nll_loss(output, target.long(), reduction='sum').item()

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq((target.long()).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    if args.task == 'rotations':
        print(100. * correct / len(test_loader.dataset))
        return 1 - correct / len(test_loader.dataset), test_loss
    elif args.task == 'pixels':
        return 0, test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Self-supervised experiments')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='depth', metavar='T',
                        help='depth/lr')
    parser.add_argument('--dataset', type=str, default='cifar', metavar='T',
                        help='dataset: cifar/mnist')
    parser.add_argument('--task', type=str, default='rotations', metavar='T',
                        help='task: rotations/pixels')
    parser.add_argument('--generalization', action='store_true', default=False,
                        help='')
    parser.add_argument('--model_type', type=str, default='hyper', metavar='M',
                        help='model type')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    print (args.dataset)
    print (args.task)
    print (args.model_type)

    if args.dataset == 'mnist':
        nc = 1
        h = 28
        dataset_class = MNIST_ROTATE
        act = nn.ReLU()
    elif args.dataset == 'cifar':
        nc = 3
        h = 32
        dataset_class = CIFAR_ROTATE
        act = nn.ELU()

    input_dim1 = input_dim2 = nc*h**2

    train_set = dataset_class(root='./data', train=True, download=True)
    test_set = dataset_class(root='./data', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                    batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                    batch_size=args.test_batch_size, shuffle=True, **kwargs)

    rates_li_test = []
    losses_li = []


    for i in range(100):

        rates_test = []
        losses = []

        for j in range(2,10):

            if args.experiment == 'depth':
                depth = j
                lr = args.lr
            elif args.experiment == 'lr':
                depth = 4
                lr = 0.0001*(10**(j/4))

            if args.model_type == 'hyper':
                model = Hypernet(args, input_dim1 = input_dim1, input_dim2 = input_dim2,
                                 hidden_dim=100, depth=depth, act=act).to(device)
            elif args.model_type == 'emb':
                model = Embedding(args, depth, input_dim1 = input_dim1, input_dim2 = input_dim2,
                                  hidden_dim=100, act=act).to(device)

            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                print (j)
                rate_test, loss = test(args, model, device, test_loader)

            rates_test += [rate_test]
            losses += [loss]

        rates_li_test += [rates_test]
        losses_li += [losses]

        rates_runs_test = np.array(rates_li_test)
        rates_means_test = rates_runs_test.mean(axis=0)
        rates_stds_test = rates_runs_test.std(axis=0)

        print ('means and std: rates_test')
        print (rates_means_test.tolist())
        print (rates_stds_test.tolist())

        losses_runs = np.array(losses_li)
        losses_means = losses_runs.mean(axis=0)
        losses_stds = losses_runs.std(axis=0)

        print('means and std: loss')
        print(losses_means)
        print(losses_stds)


if __name__ == '__main__':
    main()