<br>
<h1 align="center">读光 OCR ONNX</h1>
<br>
<p align="center">
<a href="https://github.com/xxx/xxx/blob/master/README_en_US.md">中文</a> &nbsp;|&nbsp; <a href="">English</a>
</p>

## ✨简介

本项目旨在提供一个简单易用的读光OCR ONNX 模型解决方案，以便快速上手并集成读光OCR模型到自己的项目中。

[读光OCR](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)是一款功能强大的光学字符识别模型，支持中文、英文识别。采用ONNX格式，我们可以更方便地进行部署和推理。


## 🛠️ 使用

### 环境安装

```python
onnx
```

> 测试时使用的依赖具体版本可以参考 `requirements.txt` 文件
>
> 目前只测试了cpu环境



### 模型下载

需要下载下面表格中一对文字识别和检测模型。


文字识别模型

模型大小  模型原始仓库  modelscope  百度网盘下载  huggingface

large(73.2MB)  [地址](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)  xxx  xxx

small(7.4MB)  [地址](https://modelscope.cn/models/iic/cv_LightweightEdge_ocr-recognitoin-general_damo/summary)


文字检查模型

模型大小  模型原始仓库  modelscope  百度网盘下载  huggingface

large(46.4MB)  [地址](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary)

small(5.2MB)  [地址](https://www.modelscope.cn/models/iic/cv_proxylessnas_ocr-detection-db-line-level_damo/summary)

### 使用示例



## 📍测试

速度




## 参考

[读光-文字识别-行识别模型-中英-通用领域](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)

[读光-文字检测-DBNet行检测模型-中英-通用领域](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[modelscope](https://github.com/modelscope/modelscope)
