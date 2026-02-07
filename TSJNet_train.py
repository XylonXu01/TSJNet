'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss_vif import fusion_loss_vif
import torch.nn.functional as F
# from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import time
import os
import datetime
import transforms
from Dataset import VOCDataSet
import torchvision
import torchvision.transforms as torch_transforms
from torch.cuda.amp import autocast, GradScaler
import wandb
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
# # cuda
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Creating a Gradient Scaler Example
scaler = GradScaler()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# torch.autograd.set_detect_anomaly(True)
to_tensor = torch_transforms.ToTensor()

num_epochs = 100 # total epoch
epoch_gap = -1
lr = 0.0001 # initial learning rate 
weight_decay = 0.01             # L2 regularization 0.01
batch_size = 2
# Coefficients of the loss function
coeff_mse_loss_VF = 6 # alpha1
coeff_mse_loss_IF = 1
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv1 = 5.
coeff_ssim_mse = 5.
att_weight=0.1
coeff_tv = 5.
clip_grad_norm_value = 1
optim_step = 10
optim_gamma = 0.5
# alpha=1.
# beta=20.
theta=5.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ir_VOC_root = './data/MSRS'
vi_VOC_root = './data/MSRS'
num_classes = 6

prev_time = time.time()
gray = False 
prtrain = False 

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="TSJNet-fusion",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "Fusion",
    "dataset": "MSRS",
    "epochs": 100,
    }
)


'''
------------------------------------------------------------------------------
dataloaders
------------------------------------------------------------------------------
'''

data_transform = {
    "train": transforms.Compose([# transforms.resize((512, 384)), # 将图像resize成128*128
                                transforms.ToTensor()
                                ]),
    "val": transforms.Compose([transforms.ToTensor(),
                               # transforms.resize((512, 384))
                               ])
}

train_dataset = VOCDataSet(ir_VOC_root,data_transform["train"], "train.txt",gray=gray)

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

train_data_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                num_workers=32,
                                shuffle=True,
                                collate_fn=train_dataset.collate_fn,
                                drop_last=True
                                )


loader = {'train': train_dataset, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M") # 月-日-时-分

'''
------------------------------------------------------------------------------
funsion model
------------------------------------------------------------------------------
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIDF_Encoder = Restormer_Encoder().to(device)
DIDF_Decoder = Restormer_Decoder(128, 512, 32).to(device)
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
DetailFuseLayer = DetailFeatureExtraction(n_feat = 64).to(device)



if prtrain:
    ckpt_path="./weights/10-31-08-59__epoch4.pth"
    # load the pre-trained model
    DIDF_Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    DIDF_Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])



# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)


scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=3, verbose=True)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=3, verbose=True)
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.1, patience=3, verbose=True)
scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='min', factor=0.1, patience=3, verbose=True)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
# Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

'''
------------------------------------------------------------------------------
Detction model
------------------------------------------------------------------------------
'''
ir_model = fasterrcnn_resnet50_fpn(pretrained=True)# , num_classes=7)
vi_model = fasterrcnn_resnet50_fpn(pretrained=True)# , num_classes=7)
# Get the number of input features for the model's classifier
ir_in_features = ir_model.roi_heads.box_predictor.cls_score.in_features
vi_in_features = vi_model.roi_heads.box_predictor.cls_score.in_features
# Replace the original classifier with a new one
# +1 because we have to consider the background category
ir_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(ir_in_features, 90 + 1)
vi_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(vi_in_features, 90 + 1)

ir_model.to(device)
vi_model.to(device)

ir_model.train()
vi_model.train()

'''
------------------------------------------------------------------------------
segment model
------------------------------------------------------------------------------
'''
fusion_seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)
fusion_seg_optimizer = torch.optim.SGD(fusion_seg_model.parameters(), lr=0.001, momentum=0.9)
fusion_seg_model.to(device)

fusion_seg_model.train()

criterion = torch.nn.CrossEntropyLoss()
'''
------------------------------------------------------------------------------
train
------------------------------------------------------------------------------
'''
for epoch in range(num_epochs):
    start_time = time.time()  # Start time of this epoch
    total_loss = 0.0  # Used to accumulate the loss value for each batch

    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
    for i,(data_IR, data_VIS,targets,segmentation_target)in progress_bar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        # Object Detection Model
        ir_images = [ir_image.to(device) for ir_image in data_IR]
        vi_images = [vi_image.to(device) for vi_image in data_VIS]
        ir_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        vi_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Forward propagation using the autocast context, which enables mixed precision
        with autocast():
            
            ir_loss_dict = ir_model(ir_images, ir_targets)
            vi_loss_dict = vi_model(vi_images, vi_targets)
            # Removal of categorization losses
            ir_loss_dict = {'loss_box_reg': ir_loss_dict['loss_box_reg'], 'loss_objectness': ir_loss_dict['loss_objectness'], 'loss_rpn_box_reg': ir_loss_dict['loss_rpn_box_reg']}
            vi_loss_dict = {'loss_box_reg': vi_loss_dict['loss_box_reg'], 'loss_objectness': vi_loss_dict['loss_objectness'], 'loss_rpn_box_reg': vi_loss_dict['loss_rpn_box_reg']}
            ir_det_losses = sum(loss for loss in ir_loss_dict.values())
            vi_det_losses = sum(loss for loss in vi_loss_dict.values())

            # Processing of fused data
            data_IR = torch.stack(data_IR, dim=0)  # Splice the tensor in the list along the new dimension
            data_VIS = torch.stack(data_VIS, dim=0)
            # Turn the shape of data_IR into (batch_size,1,768,1024)
            data_IR = data_IR[:,0,:,:].unsqueeze(1)
            data_VIS = data_VIS[:,0,:,:].unsqueeze(1)
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

            # Run Fusion Model
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _  = DIDF_Encoder(data_IR)
            data_VIS_hat,att_vis = DIDF_Decoder(feature_V_B, feature_V_D)
            data_IR_hat,att_ir = DIDF_Decoder( feature_I_B, feature_I_D)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D= DetailFuseLayer(feature_I_D+feature_V_D)
            data_Fuse, att = DIDF_Decoder(feature_F_B,feature_F_D)

            mse_loss_V = MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = MSELoss(data_IR, data_IR_hat)    

            fusion_loss_ = fusion_loss_vif()
            fusion_loss, loss_gradient, loss_l1, loss_SSIM = fusion_loss_(data_VIS, data_IR, data_Fuse, device)

            diverse_loss = (1 - torch.max(att, 1)[0]).mean() + att.mean()

            mse_loss_V = mse_loss_V
            mse_loss_I = mse_loss_I

            all_fusion_loss= fusion_loss+att_weight * diverse_loss+coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I

            # Segmentation Model
            fusion_seg_images = [fusion_image for fusion_image in data_Fuse]
            fusion_seg_images = torch.stack(fusion_seg_images)
            
            fusion_seg_images = fusion_seg_images.expand(-1, 3, -1, -1)
            
            tensor_list = [to_tensor(image) for image in segmentation_target]
            segmentation_target = torch.stack(tensor_list, dim=0)
            # segmentation_target:[batch_size,height, width]
            segmentation_target = segmentation_target.squeeze(1)
            # fusion_seg_targets = torch.stack(segmentation_target).to('cuda')
            fusion_seg_outputs = fusion_seg_model(fusion_seg_images)
            fusion_segmentation_target = segmentation_target.to(device)
            fusion_seg_loss = criterion(fusion_seg_outputs['out'], fusion_segmentation_target.long())

            # Running Target Detection Model for Fused Images
            fusion_dict_images = [fusion_dict_image.to(device) for fusion_dict_image in data_Fuse]
            fusion_dict_targets = ir_targets
            fusion_dict_loss = ir_model(fusion_dict_images, fusion_dict_targets)
            fusion_dict_loss = {'loss_box_reg': fusion_dict_loss['loss_box_reg'], 'loss_objectness': fusion_dict_loss['loss_objectness'], 'loss_rpn_box_reg': fusion_dict_loss['loss_rpn_box_reg']}
            fusion_det_losses = sum(loss for loss in fusion_dict_loss.values())

            all_loss = all_fusion_loss + fusion_seg_loss + fusion_det_losses + ir_det_losses + vi_det_losses



        scaler.scale(all_loss).backward()
        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        
        scaler.step(optimizer1)
        scaler.step(optimizer2)
        scaler.step(optimizer3)
        scaler.step(optimizer4)
        scaler.update()

        total_loss += all_loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)

        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time * (len(train_data_loader) / (progress_bar.n + 1) - 1)

        # Update progress bar information
        progress_bar.set_postfix(loss=avg_loss,
                                remaining_time=f"{remaining_time:.2f}s")

    # adjust the learning rate
    scheduler1.step(metrics = avg_loss)  
    scheduler2.step(metrics = avg_loss)
    scheduler3.step(metrics = avg_loss)
    scheduler4.step(metrics = avg_loss)

    wandb.log({"fusion_loss": all_fusion_loss, "fusion_seg_loss": fusion_seg_loss, "fusion_det_losses": fusion_det_losses, 
               "ir_det_losses": ir_det_losses, "vi_det_losses": vi_det_losses, "total_loss": total_loss, "avg_loss": avg_loss})
    # wandb.log({"loss": avg_loss})
    if epoch % 1 == 0:
        # save the model
        checkpoint = {
                'DIDF_Encoder': DIDF_Encoder.state_dict(),  
                'DIDF_Decoder': DIDF_Decoder.state_dict(),
                'BaseFuseLayer': BaseFuseLayer.state_dict(),
                'DetailFuseLayer': DetailFuseLayer.state_dict(),
            }
        filename = f'./weights/{timestamp}__epoch{epoch}.pth'
        torch.save(checkpoint, filename)
        print(f"\nSave checkpoint {filename} successfully!")
            
wandb.finish()





            
#-------------------------------------------------------------------------------------------------------------------------------------
        