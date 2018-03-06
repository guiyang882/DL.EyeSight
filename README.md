# 图像视频中的目标检测
本工程主要目的是集成深度学习中常用的目标检测模型，并利用目标检测模型进行图像和视频中的检测！

## 开发环境
```shell
OS: Ubuntu 16.04
Python: Python 3.6.0
Tensorflow: 1.4.1 version
Opencv: 3.2.0 version for python
```

## 实现模型介绍
* SSD+VGG
* SSD+Res
* SSD+Inception
* SSD+SqueezeNet
* SSD+Deconvolution
* YOLO

## TODOLISTS
- [x] 整理文件目录结构，按照设计模式进行
- [x] 增加数据预处理的PipeLine
    - [x] 图像插值
    - [x] 图像镜像操作(左右，上下)
    - [x] 添加随机噪声(各种模糊操作)
    - [x] 对比度拉伸
    - [x] 饱和度变化
    - [x] 图像锐化
- [x] 提高模型训练速度
    - [ ] RawData ---> TFRecords
    - [x] Single Process ---> Multi Processes
- [ ] 检测过程的可视化
- [x] 编写检测网络结构模型文件
- [x] 对数据集的处理结构的统一接口
- [x] 编写对模块的测试文件

## 使用说明

## 实验结果

## 联系我
* New Issues
* Send me E-mail: liuguiyangnwpu@163.com
