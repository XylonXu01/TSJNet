from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from img_read_save import img_save, image_read_cv2
import warnings
import logging
import cv2
from tqdm import tqdm


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


def parse_args():
    parser = argparse.ArgumentParser(description='TSJNet Inference')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id to use (e.g., 0, 1, 2)')
    parser.add_argument('--ckpt', type=str, default='./weight/TSJNet.pth', help='Path to pretrained checkpoint')
    parser.add_argument('--dataset', type=str, default='MSRS', choices=['MSRS', 'M3FD', 'RoadScene', 'LLVIP'], help='Dataset name')
    parser.add_argument('--test_folder', type=str, default='./test_img', help='Root folder of test images')
    parser.add_argument('--save_folder', type=str, default='./test_result', help='Folder to save fused results')
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (GPU: {args.gpu})")

    # ---- Load model ----
    Encoder = Restormer_Encoder().to(device)
    Decoder = Restormer_Decoder(128, 512, 32).to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction(n_feat=64).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    # ---- Inference ----
    dataset_name = args.dataset
    test_folder = os.path.join(args.test_folder, dataset_name)
    save_folder = os.path.join(args.save_folder, dataset_name)
    ir_folder = os.path.join(test_folder, 'ir')
    vi_folder = os.path.join(test_folder, 'vi')

    assert os.path.exists(ir_folder), f"IR folder not found: {ir_folder}"
    assert os.path.exists(vi_folder), f"VIS folder not found: {vi_folder}"

    img_list = sorted(os.listdir(ir_folder))
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name} | Total images: {len(img_list)}")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for img_name in tqdm(img_list, desc=f'Fusing [{dataset_name}]', unit='img'):
            data_IR = image_read_cv2(os.path.join(ir_folder, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(vi_folder, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            img = cv2.imread(os.path.join(ir_folder, img_name), 0)
            ori_shape = img.shape  # (H, W)

            data_IR = torch.FloatTensor(data_IR).to(device)
            data_VIS = torch.FloatTensor(data_VIS).to(device)

            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
            data_Fuse, _ = Decoder(feature_F_B, feature_F_D)

            # Normalize to [0, 255]
            data_Fuse = (data_Fuse - data_Fuse.min()) / (data_Fuse.max() - data_Fuse.min())
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = cv2.resize(fi, (ori_shape[1], ori_shape[0])).astype(np.uint8)

            img_save(fi, img_name.split('.')[0], save_folder)

    print(f"\nResults saved to: {save_folder}")


if __name__ == '__main__':
    main()
