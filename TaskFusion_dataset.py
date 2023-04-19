# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
from scipy.io import loadmat
import utils_image as util
from skimage import data, exposure, img_as_float
import skimage
import scipy
import math
import scipy.stats as ss
from scipy import ndimage

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = 'G:\mydataset\DST\MSRS/Visible/train/MSRS/'
            data_dir_ir = 'G:\mydataset\DST\MSRS/Infrared/train/MSRS/'
            data_dir_label = 'G:\mydataset\DST\MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor
    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1/sf)
    return x


def get_pca_matrix(x, dim_pca=15):
    """
    Args:
        x: 225x10000 matrix
        dim_pca: 15
    Returns:
        pca_matrix: 15x225
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:].T

    return pca_matrix


def show_pca(x):
    """
    x: PCA projection matrix, e.g., 15x225
    """
    for i in range(x.shape[0]):
        xc = np.reshape(x[i, :], (int(np.sqrt(x.shape[1])), -1), order="F")
        util.surf(xc)


def random_brightness(inp_img,contrast,light):
    """
    Function to randomly perturb the brightness of the input images.
    :param inp_img: A H x W x C input image.
    :return: The image with randomly perturbed brightness.
    """

    inp_img = contrast * inp_img + light

    return np.clip(inp_img, 0, 255)


def AddHaz_loop(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape

    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
class qFusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(qFusion_dataset, self).__init__()
        assert split in ['train', 'train1', 'train2','val','val1', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = 'G:\mydataset\DST\MSRS/Visible/train/MSRS/'
            data_dir_ir = 'G:\mydataset\DST\MSRS/Infrared/train/MSRS/'
            data_dir_label = 'G:\mydataset\DST\MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        elif split == 'train1':
            data_dir_vis = 'G:\mydataset\DST\MSRS/Visible/train/MSRS/'
            data_dir_ir = 'G:\mydataset\DST\MSRS/Infrared/train/MSRS/'
            data_dir_label = 'G:\mydataset\DST\MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        elif split == 'train2':
            data_dir_vis =  'C:/Users\Administrator\Desktop\M3FD_Detection\T1\Three\Fusion/'
            data_dir_ir =   'C:/Users\Administrator\Desktop\M3FD_Detection\T1\Three\Fusion/'
            data_dir_label = 'C:/Users\Administrator\Desktop\M3FD_Detection\T1\Three\G11/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        elif split == 'val1':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)



            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)

            tw = w1 + w2 + w3 + w4
            w1 = w1 / (tw)
            w2 = w2 / (tw)
            w3 = w3 / (tw)
            w4 = w4 / (tw)

            # print([w1,w2,w3,w4,(w1+w2+w3+w4)])

            in0 = np.random.choice(12, 4, False)
            in1 = in0[0]
            in2 = in0[1]
            in3 = in0[2]
            in4 = in0[3]
            # kernel = utils_sisr.anisotropic_Gaussian(ksize=9, theta=theta[0], l1=l1[0], l2=l2[0])
            # kernel3 = kernel1 * a + kernel2 * b + kernel * c
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # img_L = ndimage.filters.convolve(img_H, kernel3[..., np.newaxis], mode='wrap')  # blur
            # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
            # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel

            kernel1 = kernels[0, in1].astype(np.float64)
            kernel2 = kernels[0, in2].astype(np.float64)
            kernel3 = kernels[0, in3].astype(np.float64)
            kernel4 = kernels[0, in4].astype(np.float64)

            kernelfinal = w1 * kernel1 + w2 * kernel2 + w3 * kernel3 + w4 * kernel4
            # util.imshow(util.single2uint(kernel1[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel2[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel4[..., np.newaxis]))
            # util.imshow(util.single2uint(kernelfinal[..., np.newaxis]))
            kernelfinal = util.imresize_np(kernelfinal, scale=3 / 5)

            # k1 = np.reshape(kernelfinal, (-1), order="F")  # k.flatten(order='F')
            # # util.imshow(k1)
            # # io.savemat('k.mat', {'k': kernels})
            #
            # pca_matrix = get_pca_matrix(k1, dim_pca=15)



            P = loadmat(os.path.join('kernels', 'srmd_pca_matlab.mat'))['P']
            ta = np.reshape(kernelfinal, (-1), order="F")
            degradation_vector = np.dot(P, ta)

            k_reduced = torch.from_numpy(degradation_vector).float()

            if np.random.random() < 0.1:
                noise_level = torch.zeros(1).float()
            else:
                noise_level = torch.FloatTensor([np.random.uniform(0, 50)]) / 255.0

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)

            M = M_vector.repeat(1, 480, 640)
            # M = torch.unsqueeze(M, dim=0)  # 在第一维度添加维度
            # inp_img11 = torch.cat((torch.from_numpy(inpd).float(), M), 0)
            # IR_img11 = torch.cat((torch.from_numpy(irt).float(), M), 0)
            # T1 = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            # T2 = cv2.cvtColor(T1, cv2.COLOR_RGB2GRAY)
            img=np.array(Image.open(vis_path))


            # cv2.imshow('2', image_vis)
            if np.mean(image_vis) < 200:
                lab = cv2.cvtColor(image_vis, cv2.COLOR_BGR2YCrCb)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
                lab_planes[0] = clahe.apply(lab_planes[0])
                # lab_planes[1] = clahe.apply(lab_planes[1])
                # lab_planes[2] = clahe.apply(lab_planes[2])
                lab = cv2.merge(lab_planes)
                img = cv2.cvtColor(lab, cv2.COLOR_YCrCb2BGR)
            #
            # cv2.imshow('2',img)
            # cv2.imshow('3', image_vis)
            # cv2.waitKey(0)
            # cv2.imshow('2', image_vis)


            sf = np.random.choice([1, 2,4, 3, 3, 1,2,2, 5,1,3])

            # a= 480/sf
            # b=640/sf
            # image_vis = cv2.resize(image_vis, (int(b), int(a)), interpolation=cv2.INTER_AREA)
            # image_vis = cv2.resize(image_vis,  (int(b* sf), int(a* sf)), interpolation=cv2.INTER_AREA)
            # contrast = np.random.uniform(0.5, 1.2)
            # light = np.random.randint(-20, 10)
            # image_vis = random_brightness(image_vis, contrast, light)


            contrast = np.random.uniform(0.5, 1.5)
            image_vis = Contrast_and_Brightness(contrast, 0.2, image_vis)
            image_vis = skimage.exposure.adjust_gamma(image_vis, gamma=contrast, gain=1)
            # cv2.imshow('1',image_vis)
            # cv2.waitKey(0)

            image_vis=cv2.blur(image_vis, (sf,sf))

##########################加雾（待测试）################################
            image=image_vis
            nn = np.random.choice([0,1, 2,0,4, 5, 6, 0,7,8,0,9, 0,10,11,0,13,15,16,0,17,18,0,19,20,22])
            img_f = image / 255
            (row, col, chs) = image.shape
            A = 0.5
            # beta = 0.08
            beta = 0.01 * nn + 0.05
            size = math.sqrt(max(row, col))
            center = (row // 2, col // 2)
            foggy_image = AddHaz_loop(img_f, center, size, beta, A)
            img_f = np.clip(foggy_image * 255, 0, 255)
            image_vis=img_f
############################################################
            # cv2.imshow('2', img)
            # cv2.imshow('1',image_vis)
            # cv2.waitKey(0)
            label = np.array(Image.open(label_path))


            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            img = (
                    np.asarray(Image.fromarray(img), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            # img = np.expand_dims(img, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)

            name = self.filenames_vis[index]
            # NOIZE = torch.cat((torch.from_numpy(image_vis).float(), M), 0)
            # print(1)

            return (
                torch.tensor(image_vis),
                torch.tensor(img),
                torch.tensor(M),
                name,
                torch.tensor(label),
                torch.tensor(image_ir),
            )
        elif self.split=='train1':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]

            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            image_infi = cv2.imread(ir_path, 1)


            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)

            tw = w1 + w2 + w3 + w4
            w1 = w1 / (tw)
            w2 = w2 / (tw)
            w3 = w3 / (tw)
            w4 = w4 / (tw)

            # print([w1,w2,w3,w4,(w1+w2+w3+w4)])

            in0 = np.random.choice(12, 4, False)
            in1 = in0[0]
            in2 = in0[1]
            in3 = in0[2]
            in4 = in0[3]
            # kernel = utils_sisr.anisotropic_Gaussian(ksize=9, theta=theta[0], l1=l1[0], l2=l2[0])
            # kernel3 = kernel1 * a + kernel2 * b + kernel * c
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # img_L = ndimage.filters.convolve(img_H, kernel3[..., np.newaxis], mode='wrap')  # blur
            # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
            # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel

            kernel1 = kernels[0, in1].astype(np.float64)
            kernel2 = kernels[0, in2].astype(np.float64)
            kernel3 = kernels[0, in3].astype(np.float64)
            kernel4 = kernels[0, in4].astype(np.float64)

            kernelfinal = w1 * kernel1 + w2 * kernel2 + w3 * kernel3 + w4 * kernel4
            # util.imshow(util.single2uint(kernel1[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel2[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel4[..., np.newaxis]))
            # util.imshow(util.single2uint(kernelfinal[..., np.newaxis]))
            kernelfinal = util.imresize_np(kernelfinal, scale=3 / 5)

            # k1 = np.reshape(kernelfinal, (-1), order="F")  # k.flatten(order='F')
            # # util.imshow(k1)
            # # io.savemat('k.mat', {'k': kernels})
            #
            # pca_matrix = get_pca_matrix(k1, dim_pca=15)



            P = loadmat(os.path.join('kernels', 'srmd_pca_matlab.mat'))['P']
            ta = np.reshape(kernelfinal, (-1), order="F")
            degradation_vector = np.dot(P, ta)

            k_reduced = torch.from_numpy(degradation_vector).float()

            if np.random.random() < 0.1:
                noise_level = torch.zeros(1).float()
            else:
                noise_level = torch.FloatTensor([np.random.uniform(0, 50)]) / 255.0

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)

            M = M_vector.repeat(1,480, 640)
            # M = torch.unsqueeze(M, dim=0)  # 在第一维度添加维度
            # inp_img11 = torch.cat((torch.from_numpy(inpd).float(), M), 0)
            # IR_img11 = torch.cat((torch.from_numpy(irt).float(), M), 0)
            # T1 = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            # T2 = cv2.cvtColor(T1, cv2.COLOR_RGB2GRAY)
            img=np.array(Image.open(vis_path))


            sf = np.random.choice([22, 11, 55, 33, 9, 7, 9, 11, 5, 15, 11])
            contrast = np.random.uniform(0.4, 1.6)
            image_visf = Contrast_and_Brightness(contrast, 0.1, image_vis)
            image_visf = skimage.exposure.adjust_gamma(image_visf, gamma=contrast, gain=1)
            image_visf = cv2.blur(image_visf, (sf, sf))
            # image_visf = skimage.util.random_noise(image_visf, mode='gaussian', seed=None, clip=True)


            image_visfi = Contrast_and_Brightness(contrast, 0.1, image_infi)
            image_visfi = skimage.exposure.adjust_gamma(image_visfi, gamma=contrast, gain=1)
            image_visfi = cv2.blur(image_visfi, (sf, sf))
            # image_visfi = skimage.util.random_noise(image_visfi, mode='gaussian', seed=None, clip=True)

            hyb=(image_visf+image_visfi)*100
            # cv2.imshow('1', image_visf)
            # cv2.imshow('2', image_visfi)
            # cv2.imshow('3',hyb)
            # cv2.waitKey(0)


            # label = np.array(Image.open(label_path))

            hyb = (
                    np.asarray(Image.fromarray(hyb), dtype=np.float64).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )

            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            img = (
                    np.asarray(Image.fromarray(img), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            # img = np.expand_dims(img, axis=0)


            name = self.filenames_vis[index]
            label1 = cv2.imread(label_path, 0)
            # cv2.imshow('1', label1*250)
            # cv2.waitKey(0)
            label1 = label1[:, :, np.newaxis].transpose((2, 0, 1)) / 255.0
            label1 = label1.astype(np.float32)


            label = np.array(Image.open(label_path))
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            return (
                torch.tensor(image_vis),
                torch.tensor(img),
                torch.tensor(M),
                name,
                torch.tensor(label),
                torch.tensor(image_ir),
                torch.tensor(hyb, dtype=torch.float32)
            )
        elif self.split=='train2':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]

            image_vis = np.array(Image.open(vis_path).convert('RGB'))
            image_inf = cv2.imread(ir_path, 1)
            image_infi = cv2.imread(ir_path, 1)


            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)

            tw = w1 + w2 + w3 + w4
            w1 = w1 / (tw)
            w2 = w2 / (tw)
            w3 = w3 / (tw)
            w4 = w4 / (tw)

            # print([w1,w2,w3,w4,(w1+w2+w3+w4)])

            in0 = np.random.choice(12, 4, False)
            in1 = in0[0]
            in2 = in0[1]
            in3 = in0[2]
            in4 = in0[3]
            # kernel = utils_sisr.anisotropic_Gaussian(ksize=9, theta=theta[0], l1=l1[0], l2=l2[0])
            # kernel3 = kernel1 * a + kernel2 * b + kernel * c
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # img_L = ndimage.filters.convolve(img_H, kernel3[..., np.newaxis], mode='wrap')  # blur
            # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
            # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel

            kernel1 = kernels[0, in1].astype(np.float64)
            kernel2 = kernels[0, in2].astype(np.float64)
            kernel3 = kernels[0, in3].astype(np.float64)
            kernel4 = kernels[0, in4].astype(np.float64)

            kernelfinal = w1 * kernel1 + w2 * kernel2 + w3 * kernel3 + w4 * kernel4
            # util.imshow(util.single2uint(kernel1[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel2[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel4[..., np.newaxis]))
            # util.imshow(util.single2uint(kernelfinal[..., np.newaxis]))
            kernelfinal = util.imresize_np(kernelfinal, scale=3 / 5)

            # k1 = np.reshape(kernelfinal, (-1), order="F")  # k.flatten(order='F')
            # # util.imshow(k1)
            # # io.savemat('k.mat', {'k': kernels})
            #
            # pca_matrix = get_pca_matrix(k1, dim_pca=15)



            P = loadmat(os.path.join('kernels', 'srmd_pca_matlab.mat'))['P']
            ta = np.reshape(kernelfinal, (-1), order="F")
            degradation_vector = np.dot(P, ta)

            k_reduced = torch.from_numpy(degradation_vector).float()

            if np.random.random() < 0.1:
                noise_level = torch.zeros(1).float()
            else:
                noise_level = torch.FloatTensor([np.random.uniform(0, 50)]) / 255.0

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)

            M = M_vector.repeat(1,320, 320)
            # M = torch.unsqueeze(M, dim=0)  # 在第一维度添加维度
            # inp_img11 = torch.cat((torch.from_numpy(inpd).float(), M), 0)
            # IR_img11 = torch.cat((torch.from_numpy(irt).float(), M), 0)
            # T1 = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            # T2 = cv2.cvtColor(T1, cv2.COLOR_RGB2GRAY)
            img=np.array(Image.open(vis_path).convert('RGB'))


            sf = np.random.choice([22, 11, 55, 33, 9, 7, 9, 11, 5, 15, 11])
            contrast = np.random.uniform(0.4, 1.6)
            image_visf = Contrast_and_Brightness(contrast, 0.1, image_vis)
            image_visf = skimage.exposure.adjust_gamma(image_visf, gamma=contrast, gain=1)
            image_visf = cv2.blur(image_visf, (sf, sf))
            # image_visf = skimage.util.random_noise(image_visf, mode='gaussian', seed=None, clip=True)


            image_visfi = Contrast_and_Brightness(contrast, 0.1, image_infi)
            image_visfi = skimage.exposure.adjust_gamma(image_visfi, gamma=contrast, gain=1)
            image_visfi = cv2.blur(image_visfi, (sf, sf))
            # image_visfi = skimage.util.random_noise(image_visfi, mode='gaussian', seed=None, clip=True)


            # cv2.imshow('1', image_visf)
            # cv2.imshow('2', image_visfi)
            # cv2.imshow('3',hyb)
            # cv2.waitKey(0)

            # if target_size is not None:
            image_vis = cv2.resize(image_vis, (320, 320), interpolation=cv2.INTER_AREA)
            image_inf = cv2.resize(image_inf, (320, 320), interpolation=cv2.INTER_AREA)


            # label = np.array(Image.open(label_path))


            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )

            img = (
                    np.asarray(Image.fromarray(img), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )

            # img = np.expand_dims(img, axis=0)
            image_inf = (
                    np.asarray(Image.fromarray(image_inf), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )


            name = self.filenames_vis[index]
            label1 = cv2.imread(label_path, 0)
            label1 = cv2.resize(label1, (320, 320), interpolation=cv2.INTER_AREA)
            label1 = label1[:, :, np.newaxis].transpose((2, 0, 1)) / 255.0
            label1 = label1.astype(np.float32)

            # image_inf = image_inf[:, :, np.newaxis].transpose((2, 0, 1)) / 255.0
            # image_inf = image_inf.astype(np.float32)

            return (
                torch.tensor(image_vis),
                torch.tensor(img),
                torch.tensor(M),
                name,
                torch.tensor(label1),
                torch.tensor(image_inf),
                torch.tensor(image_inf)
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            dd = Image.open(vis_path)
            image_vis = np.array(Image.open(vis_path).convert('RGB'))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]

            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)

            tw = w1 + w2 + w3 + w4
            w1 = w1 / (tw)
            w2 = w2 / (tw)
            w3 = w3 / (tw)
            w4 = w4 / (tw)

            # print([w1,w2,w3,w4,(w1+w2+w3+w4)])

            in0 = np.random.choice(12, 4, False)
            in1 = in0[0]
            in2 = in0[1]
            in3 = in0[2]
            in4 = in0[3]
            # kernel = utils_sisr.anisotropic_Gaussian(ksize=9, theta=theta[0], l1=l1[0], l2=l2[0])
            # kernel3 = kernel1 * a + kernel2 * b + kernel * c
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # img_L = ndimage.filters.convolve(img_H, kernel3[..., np.newaxis], mode='wrap')  # blur
            # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
            # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel

            kernel1 = kernels[0, in1].astype(np.float64)
            kernel2 = kernels[0, in2].astype(np.float64)
            kernel3 = kernels[0, in3].astype(np.float64)
            kernel4 = kernels[0, in4].astype(np.float64)

            kernelfinal = w1 * kernel1 + 0 * kernel2 + 0 * kernel3 + 0 * kernel4
            # util.imshow(util.single2uint(kernel1[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel2[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel4[..., np.newaxis]))
            # util.imshow(util.single2uint(kernelfinal[..., np.newaxis]))
            kernelfinal = util.imresize_np(kernelfinal, scale=3 / 5)

            P = loadmat(os.path.join('kernels', 'srmd_pca_matlab.mat'))['P']
            ta = np.reshape(kernelfinal, (-1), order="F")
            degradation_vector = np.dot(P, ta)

            k_reduced = torch.from_numpy(degradation_vector).float()

            # if np.random.random() < 0.1:
            #     noise_level = torch.zeros(1).float()
            # else:
            #     noise_level = torch.FloatTensor([np.random.uniform(0, 50)]) / 255.0
            noise_level = torch.zeros(1).float()

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)

            M = M_vector.repeat(1, dd.height, dd.width)
            #M = image_vis.repeat(16, 480, 640)
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                M
            )
        elif self.split=='val1':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            dd = Image.open(vis_path)
            image_vis = np.array(Image.open(vis_path).convert('RGB'))
            image_inf = cv2.imread(ir_path, 1)
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            # image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            # image_ir = np.expand_dims(image_ir, axis=0)
            image_ir = (
                    np.asarray(Image.fromarray(image_inf), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )



            # image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]

            w1 = np.random.uniform(0, 1)
            w2 = np.random.uniform(0, 1)
            w3 = np.random.uniform(0, 1)
            w4 = np.random.uniform(0, 1)

            tw = w1 + w2 + w3 + w4
            w1 = w1 / (tw)
            w2 = w2 / (tw)
            w3 = w3 / (tw)
            w4 = w4 / (tw)

            # print([w1,w2,w3,w4,(w1+w2+w3+w4)])

            in0 = np.random.choice(12, 4, False)
            in1 = in0[0]
            in2 = in0[1]
            in3 = in0[2]
            in4 = in0[3]
            # kernel = utils_sisr.anisotropic_Gaussian(ksize=9, theta=theta[0], l1=l1[0], l2=l2[0])
            # kernel3 = kernel1 * a + kernel2 * b + kernel * c
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # img_L = ndimage.filters.convolve(img_H, kernel3[..., np.newaxis], mode='wrap')  # blur
            # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
            # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel

            kernel1 = kernels[0, in1].astype(np.float64)
            kernel2 = kernels[0, in2].astype(np.float64)
            kernel3 = kernels[0, in3].astype(np.float64)
            kernel4 = kernels[0, in4].astype(np.float64)

            kernelfinal = w1 * kernel1 + 0 * kernel2 + 0 * kernel3 + 0 * kernel4
            # util.imshow(util.single2uint(kernel1[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel2[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel3[..., np.newaxis]))
            # util.imshow(util.single2uint(kernel4[..., np.newaxis]))
            # util.imshow(util.single2uint(kernelfinal[..., np.newaxis]))
            kernelfinal = util.imresize_np(kernelfinal, scale=3 / 5)

            P = loadmat(os.path.join('kernels', 'srmd_pca_matlab.mat'))['P']
            ta = np.reshape(kernelfinal, (-1), order="F")
            degradation_vector = np.dot(P, ta)

            k_reduced = torch.from_numpy(degradation_vector).float()

            # if np.random.random() < 0.1:
            #     noise_level = torch.zeros(1).float()
            # else:
            #     noise_level = torch.FloatTensor([np.random.uniform(0, 50)]) / 255.0
            noise_level = torch.zeros(1).float()

            M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)

            M = M_vector.repeat(1, dd.height, dd.width)
            #M = image_vis.repeat(16, 480, 640)
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                M
            )

    def __len__(self):
        return self.length

# if __name__ == '__main__':
    # data_dir = '/data1/yjt/MFFusion/dataset/'
    # train_dataset = MF_dataset(data_dir, 'train', have_label=True)
    # print("the training dataset is length:{}".format(train_dataset.length))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # train_loader.n_iter = len(train_loader)
    # for it, (image_vis, image_ir, label) in enumerate(train_loader):
    #     if it == 5:
    #         image_vis.numpy()
    #         print(image_vis.shape)
    #         image_ir.numpy()
    #         print(image_ir.shape)
    #         break
