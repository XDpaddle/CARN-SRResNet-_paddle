# CARN   and  SRresNet

Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
[Paper] https://paperswithcode.com/paper/fast-accurate-and-lightweight-super-1

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network    
[Paper]https://arxiv.org/abs/1609.04802

ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic.
[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)


## 参考资料

- [Xiangtaokong/ClassSR](https://github.com/Xiangtaokong/ClassSR)
https://github.com/Scallions/ClassSR_paddle

Paddle 复现版本

## 数据集
### training datasets
Download the training datasets(DIV2K) and validation dataset(Set5).

Generate simple, medium, hard (class1, class2, class3) training data.

```bash
cd scripts
python generate_mod_LR_bic.py
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py
```
### testing datasets

Download the testing datasets (DIV2K_valid).

Generate simple, medium, hard (class1, class2, class3) validation data.

```bash
cd scripts
python extract_subimages_test.py
python divide_subimages_test.py
```

## 训练步骤

### train sr
```bash
python train.py -opt config/train/train_CARN.yml
python train.py -opt config/train/train_SRResNet.yml

```



## 测试步骤

```bash
python test.py -opt config/test/test_CARN.yml
python test.py -opt config/test/test_SRResNet.yml
```




## 实验结果

## 复现指标

|        | PSNR            |
| ------ | --------------- |
| Paddle | 28.17(CARN)     |
| Paddle | 29.78(SRResNet) |



