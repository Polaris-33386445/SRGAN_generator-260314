import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *

def train(train_loader, generator, discriminator, truncated_vgg19, update_ratio_g, update_ratio_d,
          content_loss_criterion, adversarial_loss_criterion, optimizer_g, optimizer_d, 
          epoch, device, beta, print_freq):
    
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses_c = AverageMeter()
    losses_a = AverageMeter()
    losses_d = AverageMeter()

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device) # [0, 1]
        hr_imgs = hr_imgs.to(device) # [0, 1]
        hr_imgs_raw = convert_image(hr_imgs, source='[0, 1]', target='[-1, 1]', device=device) # HR转化为[-1, 1]

        # GENERATOR UPDATE
        for _ in range(update_ratio_g):
            sr_imgs_raw = generator(lr_imgs) # 生成器输出格式[-1, 1]
            # 转为VGG输入格式imagenet-norm
            hr_imgs_vgg = convert_image(hr_imgs_raw, source='[-1, 1]', target='imagenet-norm', device=device)
            sr_imgs_vgg = convert_image(sr_imgs_raw, source='[-1, 1]', target='imagenet-norm', device=device)
            # VGG(i,j)特征图
            sr_feat_vgg = truncated_vgg19(sr_imgs_vgg)
            hr_feat_vgg = truncated_vgg19(hr_imgs_vgg).detach() # HR特征图不更新

            sr_disc = discriminator(sr_imgs_raw) # 判别器输入原始SR图像

            content_loss = content_loss_criterion(sr_feat_vgg, hr_feat_vgg)
            # 生成器对抗损失：希望生成的SR图像被判别器认为是真实的HR图像，标签为1
            gen_adv_loss = adversarial_loss_criterion(sr_disc, torch.ones_like(sr_disc))
            perceptual_loss = content_loss + beta * gen_adv_loss

            optimizer_g.zero_grad() # 冻结G的梯度
            perceptual_loss.backward() # 反向传播
            optimizer_g.step() # 更新G的参数

        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(gen_adv_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE
        for _ in range(update_ratio_d):
            hr_disc = discriminator(hr_imgs_raw) # 判别器输入原始HR图像
            sr_disc = discriminator(sr_imgs_raw.detach()) # 判别器更新时SR图像不更新
            # 判别器对抗损失：希望生成的SR图像被判别器认为不是真实的HR图像，标签为0；希望真实的HR图像被判别器确认，标签为1
            disc_adv_loss = adversarial_loss_criterion(sr_disc, torch.zeros_like(sr_disc)) + \
                            adversarial_loss_criterion(hr_disc, torch.ones_like(hr_disc))
            
            optimizer_d.zero_grad() # 冻结D的梯度
            disc_adv_loss.backward() # 反向传播
            optimizer_d.step() # 更新D的参数

        losses_d.update(disc_adv_loss.item(), hr_imgs.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch + 1,
                                                                          i + 1,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))

    # 释放内存
    del lr_imgs, hr_imgs, hr_imgs_raw, sr_imgs_raw, hr_imgs_vgg, sr_imgs_vgg, hr_feat_vgg, sr_feat_vgg, hr_disc, sr_disc