
<h2 align="center"><a href="">One-DM:One-Shot Diffusion Mimicker for Handwritten Text Generation</a></h2>
<div><a href=""><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> </div>
<p align="center">
  <img src="assets/js79ccvr33.png" style="width: 200px; height: 200px; margin: 0 auto;">
</p>
<!-- <a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0/"><img src="https://img.shields.io/badge/Project-Page-Green"></a> -->

<!-- <a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/Wechat-blue"></a> 
<a href="https://zhuanlan.zhihu.com/p/718163422"><img src="https://img.shields.io/badge/zhihu-red"></a>  -->

<!-- [Gang Dai](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Yifan Zhang, Quhui Ke, Qiangya Guo, Lingyu Kong, Yanming Xu,  [Zheng Ge](https://joker316701882.github.io/), Liang Zhao, [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), [Yuang Peng](https://scholar.google.com.hk/citations?user=J0ko04IAAAAJ&hl=zh-CN&oi=ao), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en) -->

<!-- <p align="center">
<img src="assets/got_logo.png" style="width: 200px" align=center>
</p> -->

## 🌟 Introduction
- We propose a One-shot Diffusion Mimicker (One-DM) for stylized handwritten text generation, which only requires a single reference sample as style input, and imitates its writing style to generate handwritten text with arbitrary content.
- Previous state-of-the-art methods struggle to accurately extract a user's handwriting style from a single sample due to their limited ability to learn styles. To address this issue, we introduce the high-frequency components of the reference sample to
 enhance the extraction of handwriting style. The proposed style-enhanced module can effectively capture the writing style patterns and suppress the interference of background noise.
- Extensive experiments on handwriting datasets in English, Chinese, and Japanese demonstrate that our approach with a single style reference even
outperforms previous methods with 15x-more references.
<div style="display: flex; flex-direction: column; align-items: center; ">
<img src="assets/overview_One-DM.png" style="width: 100%;">
</div>
<p align="center" style="margin-bottom: 10px;">
Overview of the proposed One-DM
</p>

## 🌠 Release

- [2024/9/07]🔥🔥🔥 We open-source the first version of One-DM that can generate the handwritten words. (Later versions that can support Chinese and Japanese will be released soon.)


## 🔨 Requirements
```
conda create -n One-DM python=3.8 -y
conda activate One-DM
# install all dependencies
conda env create -f environment.yml
```
## ☀️ Datasets
We provide English datasets in [Google Drive]() | [Baidu Netdisk]() PW:xu9u. Please download these datasets, uzip them and move the extracted files to /data.
## 🐳 Model Zoo
<!-- ***
- We provide the pre-trained content encoder model in [Google Drive](https://drive.google.com/drive/folders/1N-MGRnXEZmxAW-98Hz2f-o80oHrNaN_a?usp=share_link) | [Baidu Netdisk](https://pan.baidu.com/s/1RNQSRhBAEFPe2kFXsHZfLA) PW:xu9u. Please download and put it to the /model_zoo. 
- We provide the well-trained One-DM model in [Google Drive](https://drive.google.com/drive/folders/1LendizOwcNXlyY946ThS8HQ4wJX--YL7?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1RNQSRhBAEFPe2kFXsHZfLA) PW:xu9u, so that users can get rid of retraining one and play it right away.
*** -->

| Model|Google Drive|Baidu Netdisk|
|---------------|---------|-----------------------------------------|
|Pretrained One-DM|[Google Drive]()|[Baidu Netdisk PW:]()
|Pretrained OCR model|[Google Drive]()|[Baidu Netdisk PW:]()
|Pretrained Resnet18|[Google Drive]()|[Baidu Netdisk PW:]()

**Note**:
Please download these weights, and move them to /model_zoo.
## 🏋️ Training
- **training on English dataset**
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 train.py \
    --feat_model model_zoo/RN18_class_10400.pth \
    --log English
```
- **finetune on English dataset**
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_finetune.py \
    --one_dm ./Saved/IAM64_scratch/English-timestamp/model/epoch-ckpt.pt \
    --ocr_model ./model_zoo/vae_HTR138.pth --log English
 ```
**Note**:
Please modify ``timestamp`` and ``epoch`` according to your own path.

## 💡 Test
- **test on English dataset**
 ```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py \
    --one_dm ./Saved/IAM64_finetune/English-timestamp/model/epoch-ckpt.pt \
    --generate_type oov_u --dir ./Generated/English
```
**Note**:
Please modify ``timestamp`` and ``epoch`` according to your own path.
## 🍔 Exhibition
- **Comparisons with industrial image generation methods on handwritten text generation**
<p align="center">
<img src="assets/indus-English.png" style="width: 90%" align=center>
</p>

- **Comparisons with industrial image generation methods on Chinese handwriting generation**
<p align="center">
<img src="assets/indus-Chinese.png" style="width: 90%" align=center>
</p>

- **English handwritten text generation**
<p align="center">
<img src="assets/One-DM_result.png" style="width: 100%" align=center>
</p>
<!-- ![online English](assets/One-DM_result.png) -->

- **Chinese and Japanese handwriting generation**
<p align="center">
<img src="assets/casia_v4.png" style="width: 90%" align=center>
</p>
<!-- ![offline Chinese](assets/casia_v4.png) -->


## ❤️ Citation
If you find our work inspiring or use our codebase in your research, please cite our work:
```
@inproceedings{one-dm2024,
  title={One-Shot Diffusion Mimicker for Handwritten Text Generation},
  author={Dai, Gang and Zhang, Yifan and Ke, Quhui and Guo, Qiangya and Huang, Shuangping},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## ⭐ StarGraph
[![Star History Chart](https://api.star-history.com/svg?repos=dailenson/One-DM&type=Timeline)](https://star-history.com/#dailenson/One-DM&Timeline)
