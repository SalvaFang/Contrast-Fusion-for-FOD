#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class EdgeSaliencyLoss(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal
        self.ssim_loss = pytorch_msssim.msssim
        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float,requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def forward(self, y_pred):
        # Generate edge maps
        # y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))
        y_pred_edges = F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        # sal_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        # d_loss=torch.mean(torch.abs(y_pred_edges-y_gt_edges))
        #total_loss =  F.l1_loss(y_pred_edges,y_gt_edges)

        return y_pred_edges

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        # self.vgg = models.vgg16(pretrained=True).features
        #
        # self.vgg = self.vgg.cuda()
        # self.vgg.load_state_dict(torch.load('F:\PycharmProjects\lena/venv\Wangpei/vgg16.pth'))
        # for i in self.vgg.parameters():
        #     i.requires_grad_(False)
        self.mse_loss = torch.nn.MSELoss()



    def forward(self,image_vis,image_ir,labels,generate_img,i,qqout,qir):
        image_y=image_vis

        weight_zeros = torch.zeros(image_y.shape).cuda("cuda:0")
        weight_ones = torch.ones(image_y.shape).cuda("cuda:0")


        d1 = torch.where(image_y > 0.96 * weight_ones, generate_img, weight_zeros).cuda("cuda:0")  # 高光图
        d2 = torch.where(image_y > 0.96 * weight_ones, image_ir, weight_zeros).cuda("cuda:0")  # 高光图

        d3 = torch.where(image_y <= 0.96 * weight_ones, generate_img, weight_zeros).cuda("cuda:0")  # 高光图
        d4 = torch.where(image_y <= 0.96 * weight_ones, image_ir, weight_zeros).cuda("cuda:0")  # 高光图
        d5 = torch.where(image_y <= 0.96 * weight_ones, image_y, weight_zeros).cuda("cuda:0")  # 高光图

        # d2 = torch.where(image_y > 0.96 * weight_ones, torch.mean(image_y)*image_ir+(1-torch.mean(image_y))*image_y, image_y).cuda("cuda:0")  # 高光图



        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        # loss_in1 = F.l1_loss(d1, d2)
        #
        # av1 = torch.cat([image_y, image_y, image_y], 1)
        # fv3 = get_features(av1, self.vgg)
        #
        #
        # av2 = torch.cat([image_ir, image_ir, image_ir], 1)
        # fv2 = get_features(av2, self.vgg)
        #
        #
        # av4 = torch.cat([generate_img, generate_img, generate_img], 1)
        # fv4 = get_features(av4, self.vgg)

        # content_loss1 = torch.mean(F.l1_loss(fv4['conv4_3'], fv2['conv4_3']+fv3['conv4_3']))

        y_grad1 = self.sobelconv(d1)
        y_grad2 = self.sobelconv(d2)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = y_grad+ir_grad#torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)+0.2*F.l1_loss(y_grad1, y_grad2)


        # loss_grad=content_loss1


        # loss_in = (1-torch.mean(image_y)/torch.mean(x_in_max) )*F.l1_loss(image_y, generate_img)+ (torch.mean(image_y)/torch.mean(x_in_max) )*F.l1_loss(image_ir, generate_img)



        loss_total=loss_in+10*loss_grad   #光照因子作为下一篇论文
        return loss_total,loss_in,loss_in


def get_features(img, model, layers=None):
    '''获取特征层'''
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_3',
            '10': 'conv3_3',
            '19': 'conv4_4',
            '21': 'conv4_3',  # content层
            '28': 'conv5_4'
        }

    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

class Fusionlossagg(nn.Module):
    def __init__(self):
        super(Fusionlossagg, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = torch.nn.MSELoss()



    def forward(self,image_vis,image_ir,labels,generate_img,i,qqout,qir):
        image_y=image_vis[:,:1,:,:]
        weight_zeros = torch.zeros(image_y.shape).cuda("cuda:0")
        weight_ones = torch.ones(image_y.shape).cuda("cuda:0")

        # print(image_vis.max())
        # dm_tensorh = torch.where(image_vis > 0.95*weight_ones, image_ir, image_vis).cuda("cuda:0")

        # dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
        d1 = torch.where(generate_img > 0.96 * weight_ones,generate_img, weight_zeros).cuda("cuda:0")#高光图

        # d2=torch.mean(d1)*10/(torch.mean(d1)*10+1)


        y_grad = features_grad(image_y)
        ir_grad = features_grad(image_ir)
        generate_img_grad=features_grad(generate_img)
        x_grad_joint=torch.max(y_grad, ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        # y_grad = self.sobelconv(image_y)
        # ir_grad = self.sobelconv(image_ir)
        # generate_img_grad = self.sobelconv(generate_img)
        # x_grad_joint = y_grad+ ir_grad
        # loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)



        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)



        loss_total=loss_in+10*loss_grad # +10*torch.mean(torch.abs(d1)) #光照因子作为下一篇论文
        return loss_total,loss_in,loss_grad


class Fusionlossagg1(nn.Module):
    def __init__(self):
        super(Fusionlossagg1, self).__init__()

        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = pytorch_msssim.msssim


    def forward(self,image_vis,image_ir,labels,generate_img,i,qqout,qir):
        image_y=image_vis[:,:1,:,:]
        loss_total=self.mse_loss(generate_img, image_y)#+ 1-(self.ssim_loss(qqout, qir)+ 2-(self.ssim_loss(generate_img, image_vis))-(self.ssim_loss(generate_img, qqout))#+self.mse_loss(generate_img, qqout)+self.mse_loss(generate_img, qir)
        return loss_total,loss_total,loss_total


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.sobelconv = Sobelxy()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.sobelconv(x)

        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


class TVLossaggrator(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLossaggrator, self).__init__()
        self.sobelconv = Sobelxy()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        # a=[]
        # for i in range(len(x)):
        x=features_grad(x)


        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TVLossPix(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLossPix, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]

        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Fusionloss1(nn.Module):
    def __init__(self):
        super(Fusionloss1, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = pytorch_msssim.msssim


    def forward(self,image_vis,image_ir,labels,generate_img,i,qqout,qir):
        image_y=image_vis[:,:1,:,:]
        # loss_grad=F.l1_loss(generate_img,image_y)

        loss_total=self.mse_loss(generate_img, image_y)+ 1-(self.ssim_loss(generate_img, image_y))#+ 2-(self.ssim_loss(generate_img, image_vis))-(self.ssim_loss(generate_img, qqout))#+self.mse_loss(generate_img, qqout)+self.mse_loss(generate_img, qir)
        return loss_total,loss_total,loss_total

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

if __name__ == '__main__':
    pass

