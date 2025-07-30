import torch
import torch.nn.functional as F
import torch.nn as nn
from taming.modules.losses.vqperceptual import adopt_weight, VQLPIPSWithDiscriminator, VQLPIPSWithDiscriminatorEntropyLoss


class LogitLaplaceLoss(nn.Module):
    def __init__(self):
        super(LogitLaplaceLoss, self).__init__()

    def forward(self, inputs, targets):
        # Apply sigmoid to input logits to get probabilities

        probs_inputs = torch.sigmoid(inputs)
        probs_targets = torch.sigmoid(targets)
        # Safe transformation from probabilities to logits
        logits = torch.log(probs_inputs / (1.0 - probs_inputs + 1e-6) + 1e-6)
        targets = torch.log(probs_targets / (1.0 - probs_targets + 1e-6) + 1e-6)
        # Calculate Laplace loss
        loss = torch.mean(torch.abs(logits - targets))
        return loss

class VQLPIPSWithDiscriminatorEntropyDINOLoss(VQLPIPSWithDiscriminator):
    def __init__(self, disc_start, 
                    codebook_weight=1, 
                    pixelloss_weight=1,
                    entropy_loss_weight=1, 
                    dino_loss_weight=0.1,
                    disc_num_layers=3, 
                    disc_in_channels=3, 
                    disc_factor=1, 
                    disc_weight=1,
                    adaptive_disc_weight=True,
                    l1_loss_weight=1,
                    l2_loss_weight=0, 
                    perceptual_weight=1, 
                    use_actnorm=False, 
                    disc_conditional=False, 
                    disc_ndf=64, 
                    disc_loss="hinge", 
                    warmup_steps=1000, 
                    min_lr_multiplier=0.1,
                    beta_1=0.5, 
                    beta_2=0.9,):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, disc_num_layers, disc_in_channels, disc_factor, disc_weight, adaptive_disc_weight,
                          l1_loss_weight, l2_loss_weight, perceptual_weight, use_actnorm, disc_conditional, disc_ndf, disc_loss, warmup_steps, min_lr_multiplier, beta_1, beta_2)
        self.entropy_loss_weight = entropy_loss_weight
        self.dino_loss_weight = dino_loss_weight
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def forward(self, codebook_entropy_losses, dino_loss, inputs, reconstructions, optimizer_idx,
                    global_step, last_layer=None, cond=None, split="train"):
        codebook_loss, entropy_loss = codebook_entropy_losses

        laplace_loss = LogitLaplaceLoss()
        # now the GAN part
        if optimizer_idx == 0:
            #inputs = F.interpolate(inputs, size=(288, 288), mode='bilinear', align_corners=False)
            l1_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            # l2 reconstruction loss
            l2_loss = F.mse_loss(reconstructions.contiguous(), inputs.contiguous())
            rec_loss = self.l1_loss_weight * l1_loss  + self.l2_loss_weight * l2_loss

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss #+ 0.1 * laplace_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = torch.mean(nll_loss)
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.adaptive_disc_weight:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = self.disc_weight

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = nll_loss + d_weight * disc_factor * g_loss
            loss += self.codebook_weight * codebook_loss.mean()
            if entropy_loss is not None:
                loss += self.entropy_loss_weight * entropy_loss.mean()
            

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight,
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            
            if entropy_loss is not None:
                log["{}/entropy_loss_weight".format(split)] = torch.tensor(self.entropy_loss_weight)
                log["{}/entropy_loss".format(split)] = entropy_loss.detach().mean()
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminatorDino(VQLPIPSWithDiscriminator):
    def __init__(self, *args, dino_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dino_loss_weight = dino_loss_weight

    def forward(self, codebook_loss, dino_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        
        # now the GAN part
        if optimizer_idx == 0:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = torch.mean(nll_loss)
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + self.dino_loss_weight*dino_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
