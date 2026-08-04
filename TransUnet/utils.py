import numpy as np
import torch
from medpy import metric
import cv2
import torch.nn as nn
import SimpleITK as sitk
np.bool = np.bool_


# Annotation-guided binary cross entropy loss (AG-BCE)
def attention_BCE_loss(h_W, y_true, y_pred, y_std, ks = 5):
    number_of_pixels = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]

    y_true_np = y_true.cpu().detach().numpy()
    y_std_np = y_std.cpu().detach().numpy()


    hard = cv2.bitwise_xor(y_true_np, y_std_np)
    hard = hard.astype(np.uint8)
    
    # Apply dilation operation to hard regions
    kernel_size = ks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(hard.shape[0]):
        hard[i] = cv2.dilate(hard[i], kernel)
    hard = hard.astype(np.float32)

    easy = abs(hard-1)
    hard = torch.tensor(hard).cuda()
    easy = torch.tensor(easy).cuda()

    epsilon = 0.000001
    beta = 0.5

    loss = -beta*torch.mul(y_true,torch.log(y_pred + epsilon)) - (1.0 - beta)*torch.mul(1.0-y_true,torch.log(1.0 - y_pred + epsilon))
    hard_loss = torch.sum(torch.mul(loss,hard))
    easy_loss = torch.sum(torch.mul(loss,easy))

    LOSS = ((1/(1+h_W))*easy_loss + (h_W/(1+h_W))*hard_loss)/(number_of_pixels)

    return LOSS


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    hd95 = 0
    dice = 0
    num = 0

    for i in range(pred.shape[0]):
        pred_sum = pred[i,:,:].sum()
        gt_sum = gt[i,:,:].sum()
        if pred_sum>0 and gt_sum>0:
            num +=1
            dice += metric.binary.dc(pred[i,:,:], gt[i,:,:])
            hd95 += metric.binary.hd95(pred[i,:,:], gt[i,:,:])

    hd95 = (hd95*spacing)/num
    dice = dice/num

    return dice, hd95


def test_single_volume(image, label, net, spacing, origin, direction, patch_size=(224, 224)):
    """
    image: torch tensor [num_slices, H_full, W_full]  (e.g., [6, 833, 1372])
    label: torch tensor [H_patch, W_patch]             (e.g., [256, 256])
    net:   trained PyTorch model
    Returns:
        vol_pred, vol_label, vol_image: SimpleITK images
    """

    num_slices = image.shape[0]

    H_full, W_full = image.shape[1], image.shape[2]
    prediction = np.zeros((num_slices, H_full, W_full), dtype=np.uint8)

    net.eval()
    with torch.no_grad():
        for ind in range(num_slices):
            slice_img = image[ind, :, :] / 254.0  # normalize
            # Convert to [1,1,H,W] tensor
            slice_tensor = slice_img.unsqueeze(0).unsqueeze(0).float()
            # Resize to patch_size for model if needed
            if slice_tensor.shape[-2:] != patch_size:
                slice_tensor = F.interpolate(
                    slice_tensor,
                    size=patch_size,
                    mode="bilinear",
                    align_corners=False
                )

            # Forward pass
            outputs, _, _, _ = net(slice_tensor)
            out = torch.sigmoid(outputs).squeeze()  # [H_patch, W_patch]
            # Resize prediction back to original slice size
            out_tensor = out.unsqueeze(0).unsqueeze(0)  # [1,1,H_patch,W_patch]
            out_resized = F.interpolate(
                out_tensor,
                size=(H_full, W_full),
                mode="nearest"
            ).squeeze().cpu().numpy()
            # Threshold to get binary mask
            pred_binary = (out_resized > 0.5).astype(np.uint8)
            # Save into prediction volume
            prediction[ind, :, :] = pred_binary
    # Convert to SimpleITK volumes
    if prediction.shape[0] == 1:
        prediction_2d = prediction[0]  # shape [H, W]
        vol_pred = sitk.GetImageFromArray(prediction_2d.astype(np.float32))
        vol_image = sitk.GetImageFromArray(image.cpu().detach().numpy()[0].astype(np.float32))

        for vol in [vol_pred, vol_image]:
            vol.SetSpacing(spacing)
            vol.SetOrigin(origin)
            vol.SetDirection(direction)

    else:
        vol_pred = sitk.GetImageFromArray(prediction.astype(np.float32))
        vol_image = sitk.GetImageFromArray(image.cpu().detach().numpy().astype(np.float32))

        # Set spacing, origin, direction
        for vol in [vol_pred, vol_image]:
            vol.SetSpacing(spacing)
            vol.SetOrigin(origin)
            vol.SetDirection(direction)

    return vol_pred, vol_image