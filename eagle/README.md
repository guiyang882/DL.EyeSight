## 工程的主体模块
```shell
eagle
├── README.md
├── brain                   检测算法的核心模块
├── observe                 前期数据预处理的模块
│   ├── augmentors          图像处理方法部分代码
│   │   ├── arithmetic.py
│   │   ├── blur.py
│   │   ├── color.py
│   │   └── flip.py
│   └── base                基本的处理框架的父类信息
│       ├── basebatch.py
│       ├── basetype.py
│       └── meta.py
├── parameter.py            对于随机参数的控制部分代码
└── utils.py                对于工程中各个部分的通用代码
```