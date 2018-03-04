## 工程的主体模块
```shell
eagle
├── README.md
├── brain                           检测算法的核心模块
│   ├── solver                      对于模型的处理框架
│   ├── ssd                         SSD检测模型相关文件
│   │   ├── BatchGenerator.py
│   │   ├── Layer_AnchorBoxes.py
│   │   ├── Layer_L2Normalization.py
│   │   ├── SSDBoxEncoder.py
│   │   ├── SSDLoss.py
│   │   ├── box_encode_decode_utils.py
│   │   └── models
│   │       ├── feature_base_squeezenet.py
│   │       ├── feature_base_squeezenet_512.py
│   │       └── feature_base_vgg.py
│   └── yolo                        Yolo检测模型相关文件
│       ├── BaseYoloNet.py
│       ├── TinyYoloNet.py
│       └── YoloNet.py
├── observe                         前期数据预处理的模块
│   ├── augmentors                  图像处理方法部分代码
│   │   ├── arithmetic.py
│   │   ├── blur.py
│   │   ├── color.py
│   │   └── flip.py
│   └── base                        基本的处理框架的父类信息
│       ├── basebatch.py
│       ├── basetype.py
│       └── meta.py
├── parameter.py                    对于随机参数的控制部分代码
├── trainer                         实际调用的代码的处理逻辑
│   ├── __init__.py
│   ├── ssd_squeezenet_300_trainer.py
│   ├── ssd_squeezenet_512_trainer.py
│   └── ssd_vgg_trainer.py
└── utils.py                        对于工程中各个部分的通用代码

```