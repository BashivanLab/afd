import torch
import torch.nn.functional as F


AVAILABLE_LOSSES = ["hinge", "dcgan", "wgan_gp"]


def dis_hinge(dis_fake, dis_real):
    loss = torch.mean(torch.relu(1. - dis_real)) +\
        torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake, dis_real=None):
    return -torch.mean(dis_fake)


def dis_dcgan(dis_fake, dis_real):
    loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    return loss


def gen_dcgan(dis_fake, dis_real=None):
    return torch.mean(F.softplus(-dis_fake))
    # return torch.mean(F.softplus(-dis_fake)) + torch.mean(F.softplus(dis_real))


def dis_wgan_gp(dis_fake, dis_real):
    return torch.mean(dis_fake) - torch.mean(dis_real)


def gen_wgan_gp(dis_fake, dis_real=None):
    return -torch.mean(dis_fake)
    # return -torch.mean(dis_fake) + torch.mean(dis_real)


def gradient_penalty(dis, dis_real, dis_fake, device="cpu"):
    batch_size, c = dis_real.shape
    alpha = torch.rand((batch_size, 1)).repeat(1, c).to(device)
    interpolated_images = dis_fake + (dis_real - dis_fake) * alpha
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = dis(interpolated_images, decoder_only=True)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)


def log_gradient_penalty(model, x, y, grad_eps=0, device="cpu", lognorm=False):
    x.requires_grad_(True)
    logits = model(x)

    target_prob = torch.softmax(logits, dim=-1)[torch.arange(0,x.shape[0]), y]
    gradient = torch.autograd.grad(
        inputs=x,
        outputs=target_prob,
        grad_outputs=torch.ones_like(target_prob),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    if lognorm: 
        return -torch.log(torch.mean(gradient_norm - grad_eps))
    else: 
        return torch.mean((gradient_norm - 1) ** 2)



class _Loss(torch.nn.Module):

    """GAN Loss base class.
    Args:
        loss_type (str)
        is_relativistic (bool)
    """

    def __init__(self, loss_type, is_relativistic=False):
        assert loss_type in AVAILABLE_LOSSES, "Invalid loss. Choose from {}".format(AVAILABLE_LOSSES)
        self.loss_type = loss_type
        self.is_relativistic = is_relativistic

    def _preprocess(self, dis_fake, dis_real):
        C_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
        C_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)
        return dis_fake - C_xr_tilde, dis_real - C_xf_tilde


class DisLoss(_Loss):

    """Discriminator Loss."""

    def __call__(self, dis_fake, dis_real, **kwargs):
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return dis_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return dis_dcgan(dis_fake, dis_real)
            elif self.loss_type == "wgan_gp":
                return dis_wgan_gp(dis_fake, dis_real)

        else:
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xf, d_xr)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xr) - torch.log(1.0 - D_xf)
            elif self.loss_type == "wgan_gp":
                return dis_wgan_gp(d_xf, d_xr)
            else:
                raise NotImplementedError


class GenLoss(_Loss):

    """Generator Loss."""

    def __call__(self, dis_fake, dis_real=None, **kwargs):
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return gen_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return gen_dcgan(dis_fake, dis_real)
            elif self.loss_type == "wgan_gp":
                return gen_wgan_gp(dis_fake, dis_real)
        else:
            assert dis_real is not None, "Relativistic Generator loss requires `dis_real`."
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xr, d_xf)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xf) - torch.log(1.0 - D_xr)
            elif self.loss_type == "wgan_gp":
                return dis_wgan_gp(d_xr, d_xf)
            else:
                raise NotImplementedError