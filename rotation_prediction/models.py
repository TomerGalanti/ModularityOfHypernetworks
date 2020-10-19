from __future__ import print_function
import torch
import torch.nn as nn


class Hypernet(nn.Module):
    def __init__(self, args, input_dim1, input_dim2, hidden_dim, depth, act):
        super(Hypernet, self).__init__()

        self.depth = depth
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.act = act
        self.args = args

        if args.task == 'pixels':
            self.output_dim = 3
        elif args.task == 'rotations':
            self.output_dim = 12

        self.netF = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.ReLU(),
        )

        for i in range(depth):
            self.netF = nn.Sequential(
                self.netF,
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.netF = nn.Sequential(
            self.netF,
            nn.Linear(hidden_dim, input_dim2 * 10 + 10 * self.output_dim),
        )

    def forward(self, x, y):

        x = x.view(-1,self.input_dim1)
        y = y.view(-1,self.input_dim2)

        weights = self.netF(x)
        weights = weights.unsqueeze(1)
        weights_layer1 = weights[:,:,:self.input_dim2*10].view(-1, 10, self.input_dim2)
        weights_layer2 = weights[:,:,self.input_dim2*10:].view(-1, 10, self.output_dim)

        output = self.act(torch.bmm(weights_layer1, y.view(-1,y.shape[1],1)))
        output = torch.bmm(weights_layer2.view(-1,self.output_dim,10), output)\
            .view(-1, self.output_dim)

        return output

class Embedding(nn.Module):
    def __init__(self, args, i, input_dim1, input_dim2, hidden_dim, act):
        super(Embedding, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.act = act

        self.args = args

        if self.args.task == 'pixels':
            self.output_dim = 3
        elif self.args.task == 'rotations':
            self.output_dim = 12

        if self.args.experiment == 'depth':
            emb_dim = input_dim2 * 10 + 10 * 1
            self.depth = i
        else:
            emb_dim = 10000 * (i+1)
            self.depth = 3

        self.netE = nn.Sequential(
            nn.Linear(input_dim1, hidden_dim),
            nn.ReLU(),
        )
        for i in range(self.depth):
            self.netE = nn.Sequential(
                self.netE,
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.netE = nn.Sequential(
            self.netE,
            nn.Linear(hidden_dim, emb_dim),
        )

        self.netQ = nn.Sequential(
            nn.Linear(input_dim2 + emb_dim, 10),
            self.act,
            nn.Linear(10, self.output_dim),
        )

    def forward(self, x, y):

        x = x.view(-1, self.input_dim1)
        y = y.view(-1, self.input_dim2)
        x = self.netE(x)
        output = self.netQ(torch.cat([x,y], dim=1))

        return output

