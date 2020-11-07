import torch


def discriminator_loss(real_prob, fake_prob):
    loss_real = -1. * torch.mean(torch.log(real_prob))
    loss_fake = -1. * torch.mean(torch.log(1 - fake_prob))
    return loss_fake + loss_real, loss_real, loss_fake


def generator_loss(fake_prob):
    return -1. * torch.mean(torch.log(fake_prob))
