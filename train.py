import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from model import Unet
from torch import optim
from dataset import Attribute, RoadSegDataset
from test import validate


def train(model, dataset_dir, batch_size, n_classes, epochs, lr=0.00001):
    device = "cuda"
    model.to(device)
    # device = "cuda" if torch.device.
    train_dataset = RoadSegDataset(dataset_dir, n_classes)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = RoadSegDataset(dataset_dir, n_classes)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义损失函数
    criterion = torch.nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    # 开始迭代
    for ep in range(1, epochs+1):
        # 训练模式
        model.train()
        avg_loss = 0
        for item in train_dataloader:
            image = item["img"]
            mask = item["mask"]
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = mask.to(device=device, dtype=torch.float32)

            # 预测
            pred = model(image)
            loss = criterion(pred.squeeze(1), label.float())
            avg_loss += loss

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "weights/best_model1.pth")

            # if ep % 10 == 0:
            #     dice_score = validate(model, val_dataloader)
            #     print("Validation score:", dice_score)

            loss.backward()  # 反向求导
            optimizer.step()  # 更新求导后优化的参数
        ep_loss = avg_loss / len(train_dataloader)
        print("Epoch->", ep, "Loss->", ep_loss)



if __name__ == "__main__":
    model = Unet(in_channels=3, n_classes=1)
    train_anno_dir = "./data/train"
    batch_size = 4
    n_classes = 1
    epochs = 50
    train(model, train_anno_dir, batch_size, n_classes, epochs)





