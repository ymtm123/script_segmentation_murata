# semantic_segmentation_dm
セマンティックセグメンテーションの訓練と実行

## 訓練の準備
- PNGImagesにオリジナル画像を保存
  - 現状はpngファイルを想定している
- SegmentationClassにアノテーション済み画像を保存 
  - [labelpt](https://deecode.net/?p=1493)で作成されたファイルを想定している
  - この二つのディレクトリには同じファイル名で保存する
  - ファイルの画像枚数も同じになる
- make_imagesets_file.pyを実行する
  - 訓練用ファイルと検証用ファイルをランダムに設定する
  - 現状は訓練用に80% 検証用に20%の割合で配分される
  - 割合を変更する場合は9行目の数値を変更する
    
## 訓練の実行
- train.pyの20行目を変更する
- train.pyを実行する

## 訓練の結果
- Historiesにログが保存される
- Weightsに訓練したモデルが保存される

## 予測の準備と実行と結果
- TestImagesに画像を保存する
  - 現状はpngファイルを想定している
- predict.pyの13行目を変更する
  - 訓練で保存されたWeightsディレクトリのファイルを指定する
- Predictionsに予測結果が保存される