# Fire OCR

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/FireOCR/blob/main/LICENSE)

## 一、Intro
OCR toolbox with PyTorch.


## 二、Work

### 2.1 Baseline 复现[chinese_ocr](https://github.com/YCG09/chinese_ocr) 
code version：1.0.0

(Keras version to Pytoch)
CNN Backbone + CTC Loss in Pytorch

* 测试数据：合成数据360w抽取部分——346267张（342752train3515val），字符数5990
* 环境：ubuntu18，2080ti
* 原始keras项目（tf1.15+keras2.3.0）：
	* 基本配置：e10、lr step-1-0.4、32x280、bs128、adam、lr0.0005、单通道、无数据增强
	* 参数量：Total params: 4,896,742、Trainable params: 4,889,254、Non-trainable params: 7,488
	* 资源：显存占用11G，利用率约60%速度约17min/epoch
	* 结果：e7early stop，最佳e4：loss: 2.8616 - accuracy: 0.8393 - val_loss: 3.7654 - val_accuracy: 0.8652
* pth复现baseline（pth1.8.2）
	* 基本配置：同参数，loss=mean，Xavier初始化、无数据增强
	* 参数量：Total params: 4,892,454、Trainable params: 4,892,454、Non-trainable params: 0
	* 资源：显存占用6G，利用率约40%速度约19min/epoch（全部图片加载进内存后为13min/epoch）
	* 结果：e5最佳，train 0.931 val0.913


### 2.2 优化