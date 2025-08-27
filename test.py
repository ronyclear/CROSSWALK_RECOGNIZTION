import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as F


def validate(model, val_dataloader):
    # 验证模式
    model.val()
    device = "cuda"
    dice_score = 0
    num_val_batches = len(val_dataloader)
    with torch.no_grad():

        for img, gt_label in val_dataloader:
            pred = model(img.to(device))

            mask_pred = (F.sigmoid(pred) > 0.5).float()
            dice_score += dice_coeff(mask_pred, gt_label, reduce_batch_first=False)
    # 切换到训练模式
    model.train()
    return dice_score / max(num_val_batches, 1)


def dice_coeff(pred, gt_label, reduce_batch_first, epsilon=1e-6):
    assert pred.size() == gt_label.size()

    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (pred * gt_label).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + gt_label.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input, target, multiclass = False):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target, reduce_batch_first=True)


def put_mask_to_color(mask, img):
    indexes = mask == 255
    img[indexes] = [128, 0, 0]
    return img


if __name__ == "__main__":
    from model import Unet
    import glob
    from PIL import Image
    import numpy as np

    unet = Unet(in_channels=3, n_classes=1)
    # 加载参数到模型
    pth_dir = "weights/best_model.pth"
    unet.load_state_dict(torch.load(pth_dir, map_location="cpu"))
    unet.eval()
    unet.to("cuda")

    dataset_dir = "./data/test/*.jpg"
    imgs = glob.glob(dataset_dir)
    idx = 0
    for img_dir in imgs[:10]:
        img = Image.open(img_dir)
        img = img.resize((960, 540))
        img_arr = np.asarray(img)
        img_copy = img_arr.copy()
        img = img_arr.transpose((2, 0, 1))
        img = img / 255.0

        img = torch.as_tensor(img.copy()).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to("cuda")
        result = unet(img)
        prob = torch.sigmoid(result)

        prob = prob.squeeze(0).squeeze(0)
        prob_np = prob.detach().cpu().numpy()
        mask = (prob_np > 0.5).astype(np.uint8) * 255  # 0/255
        mask_img = Image.fromarray(mask)
        print(f"Inference in {idx} image, image path->{img_dir}", )
        mask_img.save(f"./result/test_{idx}_mask.jpg")
        res = put_mask_to_color(mask, img_copy)
        cv2.imwrite(f"./result/test_{idx}_color.jpg", res)
        idx += 1







