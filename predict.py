import glob
import os
import time
from PIL import Image
import numpy as np
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import torch
from utils.dataloader import make_datapath_list, DataTransform

##########################
net_path = "./data/Weights/PSPNet_resnet101_CELoss.pth"
##########################

font_manager.fontManager.addfont("./font/ipaexg.ttf")
matplotlib.rc('font', family="IPAexGothic")

# ファイルパスリスト作成（アノテーション画像のみを使用するため）
rootpath = "./data/"
_, _, _, val_anno_list = make_datapath_list(rootpath=rootpath, extension="png")

# モデルのロード
net = torch.load(net_path)
net.eval()

# 保存先ディレクトリ
save_dir = f"{rootpath}/Predictions/{os.path.basename(net_path).split('.pth')[0]}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('ネットワーク設定完了：学習済みの重みをロードしました')

# 元画像の表示
image_file_paths = glob.glob(f"{rootpath}/TestImages/*.png")
for image_file_path in image_file_paths:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    img = Image.open(image_file_path)   # [高さ][幅][色RGB]
    img_width, img_height = img.size
    axes[0].imshow(img)

    # 前処理
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    transform = DataTransform(input_size=256, color_mean=color_mean, color_std=color_std)

    # 適当なアノテーション画像を用意し、カラーパレットの情報を抜き出す
    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅]
    p_palette = anno_class_img.getpalette()
    # 実際はanno_class_imgを使用しないので適当なアノテーション画像を入力する
    phase = "val"
    img_transformed, _ = transform(phase, img, anno_class_img)

    # 推論
    x = img_transformed.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 216, 216])
    outputs = net(x.to("cuda:0"))
    y = outputs[0]  # yのサイズはtorch.Size([1, 21, 216, 216])

    # netの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    y = y.cpu().detach().numpy()
    y = np.argmax(y, axis=0)
    predict_class_img = Image.fromarray(np.uint8(y), mode="P")
    predict_class_img = predict_class_img.resize((img_width, img_height), Image.Resampling.NEAREST)
    predict_class_img.putpalette(p_palette)
    axes[1].imshow(predict_class_img)

    # 画像を透過させて重ねる
    trans_img = Image.new('RGBA', predict_class_img.size, (0, 0, 0, 0))
    predict_class_img = predict_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換

    for x in range(img_width):
        for y in range(img_height):
            # 推論結果画像のピクセルデータを取得
            pixel = predict_class_img.getpixel((x, y))
            r, g, b, a = pixel

            # (0, 0, 0)の背景ならそのままにして透過させる
            if r == 0 and g == 0 and b == 0:
                continue
            else:
                # それ以外の色は用意した画像にピクセルを書き込む
                trans_img.putpixel((x, y), (r, g, b, 150))
                # 150は透過度の大きさを指定している

    result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    axes[2].imshow(result)
    plt.suptitle(os.path.basename(image_file_path))
    fig.savefig(f"{save_dir}/{os.path.basename(image_file_path).split('.')[0]}.png", dpi=300)
    plt.clf()
    plt.close()












