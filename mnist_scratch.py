import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

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
        return p, lp


# good
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

# good
def log_pdf(x, mu, logsigma):
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
    train_optim(x, prior)

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


def train_optim(x, prior):
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

# test_1()
# test_2()
# test_3()
# test_4()
# test_5()
test_6()


