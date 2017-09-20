import torch
from torch.autograd import Variable
import numpy as np

b = 4
k = 3
x = Variable(torch.randn(b, k), requires_grad=False)

# network
mu = Variable(torch.randn(k), requires_grad=True)
logsigma = Variable(torch.ones(k), requires_grad=True)
sigma2 = torch.exp(2*logsigma)  # good

det_sigma = torch.prod(2*np.pi*sigma2,0) # good
constant = torch.rsqrt(det_sigma)  # good

diff = x - mu  # good
exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)  # good

prob = constant * torch.exp(exponent)  



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
    exponent = -0.5 * torch.sum(diff*diff/sigma2, 1)
    return constant + exponent


print('x')
print(x)
print('mu')
print(mu)
# print('logsigma')
# print(logsigma)
# print('sigma2')
# print(sigma2)
# print('det_sigma')
# print(det_sigma)
# print('constant')
# print(constant)
print('diff')
print(diff)
print('diff * (1/sigma2)')
print(diff * (1/sigma2))
print('diff*(1/sigma2)*diff')
print(diff*(1/sigma2)*diff)
print('torch.sum(diff*(1/sigma2)*diff, 1)')
print(torch.sum(diff*(1/sigma2)*diff, 1))
print('exponent')
print(exponent)
print('prob')
print(prob)
print('logprob')
print(torch.log(prob))
print('pdf')
print(pdf(x, mu, logsigma))
print('logpdf')
print(log_pdf(x, mu, logsigma))