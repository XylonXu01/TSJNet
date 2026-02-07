import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torchvision.transforms.functional import resize
import argparse
from funsion_model.function_loss.ms_ssim import MSSSIM
import kornia.losses
from funsion_model.function_loss.loss_ssim import ssim


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 1,1,2.5], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    args = parser.parse_args()

    return args

#CDD-fusionloss
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        # return loss_total,loss_in,loss_grad
        return loss_total


class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = torch.abs(self.laplacian(im_fus, 3))

        loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        return loss_JGrad

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

def cc(img1, img2):     #SCD
    eps = torch.finfo(torch.float32).eps

    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""

    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)


    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)

    return cc.mean()

#ssim---------------------------------------------------------------------------------------------------------------------------
# 方差计算
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)


    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)


    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()

#边缘损失------------------------------------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
#----------------------------------------------------------------------------------------------------------------
EPSILON = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMILoss(nn.Module):

    def __init__(self,
                 with_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='max',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=EPSILON):

        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):

        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def rmi_loss(self, input, target):


        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):


        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):

        if self.stride == 1:
            return x

        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)

#--------------------------------------------------------------------------------------------------------------
def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss


class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)
#-----------------------------------------------------------------------
def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res

def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map_ir=torch.where(map1+mask>0, one, zero)
    map_vi= 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    return res.mean()

class SeAFusion(nn.Module):
    def __init__(self):
        super(SeAFusion, self).__init__()
        self.sobelconv=Sobelxy()      

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis# [:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad