from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from validating_assumption_1.datasets import get_data, get_dims
from validating_assumption_1.models import Net, ShallowNet
import numpy as np


def print_results(li):

    means = (np.array(li)).mean(axis=0)
    stds = (np.array(li)).std(axis=0)

    print(means.tolist())
    print(stds.tolist())

    return None

def train(args, model_gt, models, device, train_loader, optimizers, epoch):
    model_gt.eval()
    mse = nn.MSELoss()
    for i in range(len(models)):
        models[i].train()
    for batch_idx, (data, _ ) in enumerate(train_loader):
        data = data.to(device)

        for i in range(len(models)):
            optimizers[i].zero_grad()
            target = model_gt(data)
            target = target
            output = models[i](data)
            loss = mse(output, target)
            loss.backward()
            optimizers[i].step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model_gt, models, device, test_loader):
    for i in range(len(models)):
        models[i].eval()
    test_loss_nets = test_loss_gt = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for (data, _ ) in test_loader:
            data = data.to(device)

            output1 = models[0](data)
            output2 = models[1](data)
            target = model_gt(data)
            test_loss_nets += mse(output1, output2) * len(output1)
            test_loss_gt += mse(output1, target) * len(output1)

    test_loss_nets /= len(test_loader.dataset)
    test_loss_gt /= len(test_loader.dataset)

    print('\nTest set: Average loss f1 vs net: {:.4f}\n'.format(test_loss_gt))

    print('\nTest set: Average loss f1 vs f2: {:.4f}\n'.format(test_loss_nets))

    return test_loss_nets.item(), test_loss_gt.item()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Validating Assumption 1')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='uniform', metavar='D',
                        help='dataset')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    train_loader, test_loader = get_data(args, kwargs)
    nc, h = get_dims(args)

    k = 2
    model_gt = Net(nc).to(device)
    models = [ShallowNet(nc, h).to(device) for i in range(k)]
    optimizers = [optim.SGD(models[i].parameters(), lr=args.lr,
                            momentum=args.momentum) for i in range(k)]

    errors_nets_li = []
    errors_gt_li = []

    for i in range(100):
        errors_nets = []
        errors_gt = []


        for epoch in range(1, args.epochs + 1):
            train(args, model_gt, models, device, train_loader, optimizers, epoch)
            print('Test:')
            error_nets, error_gt \
                = test(args, model_gt, models, device, test_loader)

            errors_nets += [error_nets]
            errors_gt += [error_gt]


        errors_nets_li += [errors_nets]
        errors_gt_li += [errors_gt]

        print('Nets:')
        print_results(errors_nets_li)

        print('GT:')
        print_results(errors_gt_li)



if __name__ == '__main__':
    main()