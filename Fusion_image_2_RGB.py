import numpy as np
import imageio
import glob
import os
import cv2
from tqdm import tqdm

# '''
# TODO: Convert Y channel image back to RGB image
# '''

def rgb2ycbcr(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    # Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    # Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    return Y, Cb, Cr

def ycbcr2rgb(Y, Cb, Cr):
    # R = Y + 1.402 * (Cr - 128 / 255.0)
    # G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    # B = Y + 1.772 * (Cb - 128 / 255.0)
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)
    return np.concatenate([R, G, B], axis=-1)


image1 = './test_img/MSRS/vi'    # Original color image path
image2 = './test_result/MSRS'        # Y channel grayscale image path
saving_path = './test_result/MSRS_RGB'    # Save path


images_list1 = glob.glob(image1 + '/*')
images_list2 = glob.glob(image2 + '/*')

for path1, path2 in zip(tqdm(images_list1), images_list2):
    # path2 = path1.replace(image1, image2)
    path1 = path2.replace(image2, image1).replace('.png', '.jpg')
    # print('path1:', path1)
    # print('path2:', path2)
    if ('png' in path1==False) or ('png' in path2==False):
        continue
    # path1 = path2.replace(image2, image1).replace('.png', '.png')
    # 如果path1不存在，将文件后缀改为png
    if os.path.exists(path1)==False:
        path1 = path1.replace('.jpg', '.png')
    img1 = cv2.imread(path1, 1)
    # print(img1)
    Shape1 = img1.shape
    # print(Shape1, '-------------------')
    img1_1, img1_cb, img1_cr = rgb2ycbcr(img1)
    h1 = Shape1[0]
    w1 = Shape1[1]
    output = cv2.imread(path2, 0)
    Shape2 = output.shape
    # print(Shape2)
    # print(Shape1.shape)
    output_1 = ycbcr2rgb(output, img1_cb, img1_cr)
    save_path = path1.replace(image1, saving_path).replace('.jpg', '.png').replace('.tiff', '.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_1)