<br>
<h1 align="center">读光 OCR ONNX</h1>
<br>
<p align="center">
<a href="https://github.com/xxx/xxx/blob/master/README_en_US.md">中文</a>  |  <a href="">English</a>
</p>

## ✨简介

本项目旨在提供一个简单易用的读光OCR ONNX 模型解决方案，以便快速上手并集成读光OCR模型到自己的项目中。

[读光OCR](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)是一款功能强大的光学字符识别模型，支持中文、英文识别。采用ONNX格式，我们可以更方便地进行部署和推理。

![](./assets/result.png)

## 🛠️ 使用

### 环境安装

```python
onnx
onnxruntime
numpy
pyclipper
shapely
opencv-python
pillow
```

> 测试时使用的依赖具体版本可以参考 `requirements.txt` 文件
>
> 目前只测试了cpu环境

### 模型下载

需要下载下面表格中一对文字识别和检测模型。

| 模型           | 模型大小      | 模型原始仓库                                                                                                                                                                                             | 百度网盘下载                                                  | modelscope下载                                                     | 个人评价 |
| -------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------ | -------- |
| base_seglink++ | 73.2MB+78MB   | rec[地址](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)，det [地址](https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-line-level_damo/summary)               | [地址](https://pan.baidu.com/s/1HRDW2-JFnzDoMcdU560OlA?pwd=qjl8) | [地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx/summary) | 9分      |
| large          | 73.2MB+46.4MB | rec[地址](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)，det [地址](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary)        | [地址](https://pan.baidu.com/s/1BQeeOelYU0N5PJSuf_kG3A?pwd=gztj) | [地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx/summary) | 8分      |
| small          | 7.4MB+5.2MB   | rec[地址](https://modelscope.cn/models/iic/cv_LightweightEdge_ocr-recognitoin-general_damo/summary)，det [地址](https://www.modelscope.cn/models/iic/cv_proxylessnas_ocr-detection-db-line-level_damo/summary) | [地址](https://pan.baidu.com/s/1kyWRX18-5MRkizyoGz-I7Q?pwd=khkj) | [地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx/summary) | 5分      |

> rec 为文本识别模型，det为文本检测模型

### 使用示例

base_seglink++：`demo_seglink.py`

large，small：`demo.py`

打印结果说明

```
[[[73.0, 612.0], [729.0, 626.0], [728.0, 670.0], [72.0, 656.0]], ('家记忆研究院国家记忆研究院波', 0.9971)]

三部分分别是
[box, (text,score)]; box 为文本框四个点坐标, text 为识别的文本, score 为文本的置信度
```

## 📍测试

> 20张图片
>
> CPU AMD R7 7840HS (3.80 GHz) 8核16线程

**base_seglink++**

| cpu_thread_num | 平均速度（s） | 速度区间     | 峰值内存(MB) | 闲时内存(MB) |
| -------------- | ------------- | ------------ | ------------ | ------------ |
| 1              | 3.86          | [2.53, 6.37] | 512          | 243          |
| 2              | 2.28          | [1.36, 4.22] | 512          | 243          |
| 4              | 1.57          | [0.82, 3.33] | 512          | 243          |

**模型大小：large**

| cpu_thread_num | 平均速度（s） | 速度区间  | 峰值内存(MB) | 闲时内存(MB) |
| -------------- | ------------- | --------- | ------------ | ------------ |
| 1              | 3.6           | [2.2-6.9] | 976          | 219          |
| 2              | 2.07          | [1.1-4.2] | 976          | 219          |
| 4              | 1.64          | [0.8-4.2] | 976          | 219          |

**模型大小：small**

| cpu_thread_num | 平均速度（s） | 速度区间   | 峰值内存(MB) | 闲时内存(MB) |
| -------------- | ------------- | ---------- | ------------ | ------------ |
| 1              | 1.15          | [0.9-1.5]  | 560          | 118          |
| 2              | 0.85          | [0.64-1.2] | 560          | 118          |
| 4              | 0.76          | [0.57-1.1] | 560          | 118          |

## 感谢

[读光-文字识别-行识别模型-中英-通用领域](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)

[读光-文字检测-DBNet行检测模型-中英-通用领域](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[modelscope](https://github.com/modelscope/modelscope)

[ONNXRuntime CPU推理优化 ](https://rapidai.github.io/RapidOCRDocs/blog/2022/09/23/onnxruntime-cpu%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96/)
