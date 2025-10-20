from __future__ import print_function
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import matplotlib
matplotlib.use('Agg')


# torch.save(data, open('traindata.pt', 'wb'))  # (100, 1000) = (N, time)

class Sequence(nn.Module):
    def __init__(self, n_stocks=10, n_features=11, d_hidden=128, n_layers=2):
        super(Sequence, self).__init__()
        assert not d_hidden % 2, "d_input must be a multiple of 2"

        self.dim = d_hidden
        self.W_s = nn.Linear(n_stocks, d_hidden//2)
        self.W_feat = nn.Linear(n_features*n_stocks, d_hidden//2)
        self.lstm = nn.LSTM(
            input_size=d_hidden,
            hidden_size=d_hidden,
            batch_first=True,
            num_layers=2,
            dropout=0.4
        )
        self.linear = nn.Linear(d_hidden, n_stocks*(1 + n_features))

    def forward(self, stocks, features, future=2):
        outputs = []
        input = self.W_s(stocks)
        h_t = self.W_feat(features)
        h_t = torch.zeros(input.size(0), self.dim, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.dim, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.dim, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.dim, dtype=torch.double)

        ## input = (stocks,time,features)
        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()