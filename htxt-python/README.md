# htxt-python

该文件夹存放python工具类和python-app

| 目录                                | 说明                                         |
| ----------------------------------- | -------------------------------------------- |
| `./tools/benchmark`                 | 打分程序                                     |
| `./tools/image_splitter`            | 切图程序                                     |
| `./tools/common`                    | 一些工具类                                   |
| `./test_set`                        | 测试集输入的路径，有一些大小不等的图用于测试 |
| `./prediction`                      | 模型输出的路径，并没有做输出功能             |
| `./grand_truth`                     | 测试集标记的路径                             |
| `./dataset`                         | 原始数据集路径，有一张图用于测试             |
| `./dist_dataset`                    | 切图后存放路径                               |
| `./object_detection`                | 官方代码和htxt_model类                       |
| `./object_detection/exported_model` | 模型存放路径                                 |

## 环境搭建

首先，可以先用anaconda为项目搭建虚拟环境：

```bash
conda create -n htxt-python python=3.6
activate htxt-python
```

安装TensorFlow：

```bash
# For CPU
pip install tensorflow -i  https://pypi.tuna.tsinghua.edu.cn/simple/
# For GPU
pip install tensorflow-gpu i  https://pypi.tuna.tsinghua.edu.cn/simple/
```

安装相关依赖：

```bash
pip install Cython contextlib2 jupyter matplotlib pillow lxml opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## Quick start

对`detector.py`（或`benchmark.py`或`image_splitter.py`）里的路径进行修改后运行即可

## TODO

（为什么这个输出还是有些重叠的框框呢…我觉得写得没猫病呀…）