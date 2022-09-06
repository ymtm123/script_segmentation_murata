# パッケージのimport
import os
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


def make_datapath_list(rootpath, extension):
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    img_base_dir = os.path.join(rootpath, 'PNGImages')
    anno_base_dir = os.path.join(rootpath, 'SegmentationClass')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = os.path.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = os.path.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names, encoding="utf-8_sig"):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = os.path.join(img_base_dir, f"{file_id}.{extension}")  # 画像のパス
        anno_path = os.path.join(anno_base_dir, f"{file_id}.{extension}")  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names, encoding="utf-8_sig"):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = os.path.join(img_base_dir, f"{file_id}.{extension}")  # 画像のパス
        anno_path = os.path.join(anno_base_dir, f"{file_id}.{extension}")  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.9, 1.1]),  # 画像の拡大
                RandomRotation(angle=[-180, 180]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):

    def __init__(self, img_list, anno_list, phase, transform, n_classes, one_hot=False):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.n_classes = n_classes
        self.one_hot = one_hot

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        # 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)  # [高さ][幅][色RGB]
        # アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  # [高さ][幅]

        # 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        # 正解ラベル毎のOne-hotに変換
        if self.one_hot:
            masks = torch.Tensor()
            for i in range(self.n_classes):
                mask = torch.where(anno_class_img == i,
                                   torch.tensor([1], dtype=torch.uint8),
                                   torch.tensor([0], dtype=torch.uint8))
                masks = torch.cat((masks, mask[None, :, :]), dim=0)

            anno_class_img = masks

        return img, anno_class_img
