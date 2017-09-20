import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def pdf(x, mu, logsigma):
    """
        x: (b, k)
        mu: (k)
        sigma: (k)

        sigma2: diagonal entries of covariance entries
    """
    sigma2 = torch.exp(2*logsigma)
    det_sigma = torch.prod(2*np.pi*sigma2, 0)
    constant = torch.rsqrt(det_sigma)
    diff = x - mu
    exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
    return constant * torch.exp(exponent)

def log_pdf(x, mu, logsigma):
    """
        x: (b, k)
        mu: (k)
        sigma: (k)

        sigma2: diagonal entries of covariance entries
    """
    sigma2 = torch.exp(2*logsigma)
    constant = -0.5 * torch.sum(torch.log(2*np.pi*sigma2))
    diff = x - mu
    exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
    return constant + exponent

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return nn.Parameter(torch.randn(*size) * xavier_stddev, requires_grad=True)

class CF_prior(nn.Module):
    def __init__(self, dim):
        super(CF_prior, self).__init__()
        self.k = dim
        self.mu = xavier_init([self.k])
        self.logsigma = nn.Parameter(torch.ones(self.k), requires_grad=True)


    # good
    def pdf(self, x, mu, logsigma):
        """
            x: (b, k)
            mu: (k)
            sigma: (k)

            sigma2: diagonal entries of covariance entries
        """
        sigma2 = torch.exp(2*logsigma)
        det_sigma = torch.prod(2*np.pi*sigma2, 0)
        constant = torch.rsqrt(det_sigma)
        diff = x - mu
        exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
        return constant * torch.exp(exponent)

    # good
    def log_pdf(self, x, mu, logsigma):
        """
            x: (b, k)
            mu: (k)
            sigma: (k)

            sigma2: diagonal entries of covariance entries
        """
        sigma2 = torch.exp(2*logsigma)
        det_sigma = torch.prod(2*np.pi*sigma2, 0)
        constant = -0.5 * torch.log(det_sigma)
        diff = x - mu
        exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
        return constant + exponent

    def forward(self, x):
        p = self.pdf(x, self.mu, self.logsigma)
        lp = self.log_pdf(x, self.mu, self.logsigma)
        return p#, lp

class CF_likelihood(nn.Module):
    def __init__(self, dim):
        super(CF_likelihood, self).__init__()
        self.fc1 = nn.Linear(dim, dim)

    def forward(self, x):
        return F.relu(self.fc1(x))  # relu because MNIST > 0

class ProgramStep(nn.Module):
    def __init__(self, dim, k):
        super(ProgramStep, self).__init__()
        self.dim = dim
        self.k = k

        self.priors = nn.ModuleList()
        self.likelihoods = nn.ModuleList()

        for kk in range(k):
            self.priors.append(CF_prior(dim))
            self.likelihoods.append(CF_likelihood(dim))

    def forward(self, x):
        # get weights  good
        ps = [prior(x) for prior in self.priors]  # list of size k of b
        z = torch.sum(torch.stack(ps), 0)  # b
        ws = [p/z for p in ps]  # list of size k of b

        # compute next  good
        ys = [ws[i].view(-1, 1).repeat(1, self.dim)*self.likelihoods[i](x) for i in range(len(ws))]  # list of size k of (b, dim)
        y = torch.sum(torch.stack(ys), 0)  # (b, dim)
        return y

class InputEncoder(nn.Module):
    def __init__(self, outdim):
        super(InputEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, outdim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

class OutputDecoder(nn.Module):
    def __init__(self, indim):
        super(OutputDecoder, self).__init__()
        self.fc1 = nn.Linear(indim, 10)

    def forward(self, x):
        return F.log_softmax(self.fc1(x))

class Program(nn.Module):
    def __init__(self, dim, k, steps):
        super(Program, self).__init__()
        self.dim = dim
        self.k = k
        self.steps = steps
        # self.encoder = InputEncoder(dim)
        self.prog_step = ProgramStep(self.dim, self.k)
        # self.decoder = OutputDecoder(dim)

    def forward(self, x):
        for s in xrange(self.steps):
            x = self.prog_step(x)  
        return x

##########################
##########################
def test_1():
    b = 1
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    # network
    mu = Variable(torch.randn(k), requires_grad=True)
    logsigma = Variable(torch.ones(k), requires_grad=True)

    train_manual(x, mu, logsigma)

def test_2():
    b = 10
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    # network
    mu = Variable(torch.randn(k), requires_grad=True)
    logsigma = Variable(torch.ones(k), requires_grad=True)

    train_manual(x, mu, logsigma)

def test_3():
    b = 1
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    prior = CF_prior(3)
    train_nn(x, prior)

def test_4():
    b = 10
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    prior = CF_prior(3)
    train_nn(x, prior)

def test_5():
    b = 1
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    prior = CF_prior(3)
    train_optim(x, prior)

def test_6():
    b = 10
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    prior = CF_prior(3)
    train_optim_prior(x, prior)

def test_7():
    b = 10
    k = 3
    x = Variable(torch.randn(b, k), requires_grad=False)

    likelihood = CF_likelihood(k)
    train_optim_model(x, likelihood)

def test_8():
    b = 2
    k = 4
    x = Variable(torch.randn(b, k), requires_grad=False)

    prog = ProgramStep(4, 3)
    train_optim_model(x, prog)

def test_9():
    b = 2
    k = 4
    x = Variable(torch.randn(b, k), requires_grad=False)

    prog = Program(4, 3, 5)
    train_optim_model(x, prog)


def train_nn(x, prior):
    lr = 1e-2
    for t in range(10):
        print(t)
        print('mu')
        print(prior.mu)
        print('logsigma')
        print(prior.logsigma)

        # forward
        p, lp = prior(x)
        loss = -torch.sum(lp,0)
        print('p: {}, lp: {}, loss: {}'.format(p.data[0], lp.data[0], loss.data[0]))

        # backward
        prior.zero_grad()
        loss.backward()
        for param in prior.parameters():
            param.data -= lr * param.grad.data
        print('#'*80) 

def train_optim_prior(x, prior):
    lr = 1e-2
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    for t in range(10):
        print(t)
        print('mu')
        print(prior.mu)
        print('logsigma')
        print(prior.logsigma)

        # forward
        p, lp = prior(x)
        loss = -torch.sum(lp,0)
        print('p: {}, lp: {}, loss: {}'.format(p.data[0], lp.data[0], loss.data[0]))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('#'*80) 

def train_manual(x, mu, logsigma):
    lr = 1e-2
    for t in range(1000):
        print(t)
        print('mu')
        print(mu)
        print('logsigma')
        print(logsigma)

        # forward
        p = pdf(x, mu, logsigma)
        lp = log_pdf(x, mu, logsigma)
        loss = -lp.sum()
        print('p: {}, lp: {}, loss: {}'.format(p.data[0], lp.data[0], loss.data[0]))

        # backward
        if t > 1:
            mu.grad.data.zero_()
            logsigma.data.zero_()
        loss.backward()

        mu.data -= lr * mu.grad.data
        logsigma.data -= lr * logsigma.grad.data
        print('#'*80)

def train_optim_model(x, model):
    lr = 3e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    for t in range(1000):
        # forward
        y = model(x)
        loss = loss_function(y, x)
        print('loss: {}'.format(loss.data[0]))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('#'*80) 


test_1()
# test_2()
# test_3()
# test_4()
# test_5()
# test_6()
# test_7()
# test_8()
# test_9()

