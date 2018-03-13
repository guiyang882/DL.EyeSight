## 工程的主体模块
```shell
eagle
├── brain
│   ├── rotation
│   │   └── yolo
│   ├── solver
│   │   ├── solver.py
│   │   ├── ssd_solver.py
│   │   └── yolo_solver.py
│   ├── ssd
│   │   ├── anchor_boxes.py
│   │   ├── box_encode_decode_utils.py
│   │   ├── loss.py
│   │   ├── models
│   │   │   ├── components.py
│   │   │   ├── net.py
│   │   │   ├── squeezenet_300.py
│   │   │   ├── squeezenet_512.py
│   │   │   └── vgg.py
│   │   └── normalization.py
│   └── yolo
│       ├── net.py
│       ├── yolo_net.py
│       └── yolo_tiny_net.py
├── observe
│   ├── augmentors
│   │   ├── arithmetic.py
│   │   ├── blur.py
│   │   ├── color.py
│   │   └── flip.py
│   └── base
├── parameter.py
└── utils.py

eagle
├── README.md
├── brain                           检测算法的核心模块
│   ├── solver                      对于模型的处理框架
│   ├── ssd                         SSD检测模型相关文件
│   │   └── models
│   └── yolo                        Yolo检测模型相关文件
├── observe                         前期数据预处理的模块
│   ├── augmentors                  图像处理方法部分代码
│   └── base                        基本的处理框架的父类信息
├── parameter.py                    对于随机参数的控制部分代码
├── trainer                         实际调用的代码的处理逻辑
└── utils.py                        对于工程中各个部分的通用代码

```