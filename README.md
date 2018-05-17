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

## 运行说明
1. 先准备好数据集
```shell
cat /Volumes/projects/DataSets/VOC2007/voc_train.txt
image_path01 xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id
image_path02 xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id
image_path03 xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id

PS：Class_id从0开始编号，顺序同cfg文件中的label顺序一致
```
2. 修改配置文件
配置文件存放在根目录下：**conf/ssd_train.cfg**
其中还有若干配置项，进行修改

3. 运行程序
进入到example/ssd目录中
```
python vgg_trainer.py -c ../../conf/ssd_train.cfg
```

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

## 实验结果
- YOLOv1模型在Pascal VOC数据集上的表现
<table border="0" align="center" cellpadding="0" cellspacing="0">
  <tr>
    <td valign="top">
        <div style="margin-left:100px;">
            <img src="https://github.com/liuguiyangnwpu/DL.EyeSight/blob/master/results/test_res_image/loss1.png" width="300" height="120"/>
        </div>
    </td>
    <td valign="top">
        <div style="margin-left:100px;">
            <img src="https://github.com/liuguiyangnwpu/DL.EyeSight/blob/master/results/test_res_image/loss2.png" width="300" height="120"/>
        </div>
    </td>
    <td valign="top">
        <div style="margin-left:100px;">
            <img src="https://github.com/liuguiyangnwpu/DL.EyeSight/blob/master/results/test_res_image/loss3.png" width="300" height="120"/>
        </div>
    </td>
  </tr>
</table>


## 联系我
* New Issues
* Send me E-mail: liuguiyangnwpu@163.com
