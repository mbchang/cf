import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        # print('x')
        # print(x)
        return F.log_softmax(self.fc1(x))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return nn.Parameter(torch.randn(*size) * xavier_stddev, requires_grad=True)

class CF_prior(nn.Module):
    def __init__(self, dim):
        super(CF_prior, self).__init__()
        self.k = dim
        self.mu = xavier_init([self.k])
        self.logsigma = nn.Parameter(torch.FloatTensor(self.k).fill_(0.1), requires_grad=True)


    # good
    def pdf(self, x, mu, logsigma):
        """
            x: (b, k)
            mu: (k)
            sigma: (k)

            sigma2: diagonal entries of covariance entries
        """
        sigma2 = torch.exp(2*logsigma)
        det_sigma = torch.prod(2*np.pi*sigma2)  # this becomes very huge
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
        # OK WE ACTUALLY HAVE TO DO THE MAX THING HERE!

        sigma2 = torch.exp(2*logsigma)
        constant = -0.5 * torch.sum(torch.log(2*np.pi*sigma2))
        diff = x - mu
        exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
        # both mu and logsigma become nan for some reason
        # print('mu')
        # print(mu)
        # print('logsigma')
        # print(logsigma)
        # print('constant')
        # print(constant)
        # print('exponent')
        # print(exponent)
        return constant + exponent

    def forward(self, x):
        # p = self.pdf(x, self.mu, self.logsigma)
        lp = self.log_pdf(x, self.mu, self.logsigma)  # PROBLEM basically this thing gets very very negative.
        # # print(lp)
        # # assert False
        # # subtract max
        # maxlp = torch.max(lp)  # don't do this because this is over the batch?  # okay, we are getting 
        # # print('maxlp')
        # # print(maxlp.data[0])
        # lp -= maxlp
        # print('lp')
        # print(lp)
        # # exponentiate
        # p = torch.exp(lp)
        # print('p before renormalize')
        # print(p)
        # # renormalize
        # z = torch.sum(p)  # this should not be over the batch!  # you shoudln't normalize acoss x's!
        # print('z prior')
        # print(z)
        # p = torch.div(p, z)
        # print('p after renomralize')
        # print(p)
        return lp#, lp

class CF_likelihood(nn.Module):
    def __init__(self, dim):
        super(CF_likelihood, self).__init__()
        self.fc1 = nn.Linear(dim, dim)

    def forward(self, x):
        return F.sigmoid(self.fc1(x))  # relu because MNIST > 0

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
        lps = [prior(x) for prior in self.priors]  # list of size k of b

        # here you have the log probabiities
        # print('lps yo')
        # print(lps)
        lps = torch.stack(lps)
        # print('stack')
        # print(lps)
        lps_max = torch.max(lps, 0)[0]
        # print('lps max')
        # print(lps_max)
        lps -= lps_max
        # print('subtract')
        # print(lps)
        ps = torch.exp(lps)
        # print('exp')
        # print(ps)
        z = torch.sum(ps, 0)
        # print('z')
        # print(z)
        # assert False


        # z = torch.sum(torch.stack(ps), 0)  # b
        # print('z')
        # print(z)
        ws = [p/z for p in ps]  # list of size k of b
        # print('ws')
        # print(ws)

        # assert False
        # compute next  good
        ys = [ws[i].view(-1, 1).repeat(1, self.dim)*self.likelihoods[i](x) for i in range(len(ws))]  # list of size k of (b, dim)
        y = torch.sum(torch.stack(ys), 0)  # (b, dim)
        return y

class Program(nn.Module):
    def __init__(self, dim, k, steps):
        super(Program, self).__init__()
        self.dim = dim
        self.k = k
        self.steps = steps
        self.encoder = InputEncoder(self.dim)
        self.prog_step = ProgramStep(self.dim, self.k)
        self.decoder = OutputDecoder(self.dim)

    def forward(self, x):
        x = self.encoder(x)
        for s in xrange(self.steps):
            x = self.prog_step(x)
        x = self.decoder(x)
        return x

