# pytorchのインストールは下のURLから
# https://pytorch.org/get-started/previous-versions/

import sys
import random
import math
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
import segmentation_models_pytorch as smp

###################################
CLASSES = ["back", "weldline"]  # クラスを登録する
###################################

# 初期設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.cuda.init()

n_classes = len(CLASSES)

# ファイルパスリスト作成
rootpath = "./data/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath, extension="png")

# Dataset作成
# (RGB)の色の平均値と標準偏差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                           transform=DataTransform(input_size=256, color_mean=color_mean, color_std=color_std),
                           n_classes=n_classes, one_hot=True)
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                         transform=DataTransform(input_size=256, color_mean=color_mean, color_std=color_std),
                         n_classes=n_classes, one_hot=True)

# DataLoader作成
batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# モデルを設定する
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

net = smp.PSPNet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=n_classes,
    activation=ACTIVATION,
)

print('ネットワーク設定完了：学習済みの重みをロードしました')


# 損失関数
class CELoss(nn.Module):
    def __init__(self, weights=None):
        super(CELoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets.argmax(dim=1), self.weights.to("cuda:0"), reduction="sum")
        return loss


criterion = CELoss(weights=torch.Tensor([1., 100.]))

optimizer = torch.optim.Adam([dict(params=net.parameters(), lr=0.0001)])

# スケジューラーの設定
def lambda_epoch(epoch):
    max_epoch = 10000
    return math.pow((1 - epoch / max_epoch), 0.9)


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# ネットワークをGPUへ
net.to(device)

# ネットワークがある程度固定であれば高速化させる
torch.backends.cudnn.benchmark = True

# 画像の枚数
num_train_imgs = len(dataloaders_dict["train"].dataset)
num_val_imgs = len(dataloaders_dict["val"].dataset)

# ログファイルの作成
with open(f'{rootpath}/Histories/{net._get_name()}_{ENCODER}_{criterion._get_name()}.txt', 'w') as f:
    title_list = ["epoch", "\t", "train_loss", "\t", "test_loss", "\n"]
    f.writelines(title_list)

# 学習
num_epochs = 10000
min_loss = 1e+10
for epoch in range(num_epochs):

    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和

    # epochごとの訓練と検証のループ
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()  # モデルを訓練モードに
        else:
            net.eval()  # モデルを検証モードに

        # データローダーからmini_batchずつ取り出すループ
        for imges, anno_class_imges in dataloaders_dict[phase]:
            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device)
            anno_class_imges = anno_class_imges.to(device)

            # optimizerを初期化
            optimizer.zero_grad()
            # 順伝搬（forward）計算
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(imges)
                loss = criterion(outputs, anno_class_imges) / len(imges)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item() * len(imges)
                else:
                    epoch_val_loss += loss.item() * len(imges)

    # epochのphaseごとのlossと正解率
    print('epoch {:05d}/{:05d} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Loss:{:.4f}'.format(
        epoch + 1, num_epochs, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs))

    # Lossの最小値が更新されればモデルを保存
    if epoch_val_loss / num_val_imgs < min_loss:
        min_loss = epoch_val_loss / num_val_imgs
        torch.save(net, f"{rootpath}/Weights/{net._get_name()}_{ENCODER}_{criterion._get_name()}.pth")

    # 学習履歴の保存
    log_list = ["{:06d}".format(epoch + 1), "\t",
                "{:.08f}".format(epoch_train_loss / num_train_imgs), "\t",
                "{:.08f}".format(epoch_val_loss / num_val_imgs), "\n"]
    with open(f'{rootpath}/Histories/{net._get_name()}_{ENCODER}_{criterion._get_name()}.txt', 'a') as f:
        f.writelines(log_list)
