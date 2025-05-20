<br>
<h1 align="center">读光 OCR ONNX</h1>
<br>

## 进展🎉

- [2025/5]  支持【批量】和【GPU】推理。为 `large`和 `base_seglink++` 的det模型新增 `512x512`, `1024x1024` 等多种尺寸的版本

## ✨简介

本项目是一个用于[读光OCR](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)模型的ONNX格式推理方案，帮助快速集成高性能 `中英文`OCR功能。

![](./assets/result.png)

## 主要特性

- 🚀 支持读光OCR模型的ONNX格式推理
- ⚡ 支持批量和GPU推理
- 🔍 支持多种图片格式输入（jpg, png）
- 📦 简单易用的API接口

## 🛠️ 使用

### 安装依赖

【1】CPU

采用 pip 直接安装下面依赖即可

```python
onnx
onnxruntime
numpy==1.26.3
pyclipper
shapely
opencv-python
pillow
```

> 测试时使用的依赖具体版本可以参考 `requirements.txt` 文件

【2】GPU

测试所用依赖：requirements_gpu.txt

除了 `onnxruntime-gpu` ，其他依赖均可以通过pip直接安装。

因为国内通过pip直接安装最新的 `onnxruntime-gpu` 默认只支持 `cuda12.x` 。如果想安装支持 `cuda 11.x` 版本，需要使用onnxruntime提供的源（[官方文档](https://onnxruntime.ai/docs/install/#python-installs) ）

```python
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

> 注意，这可能需要 `魔法`上网，也可以尝试在网页[onnxruntime-gpu 1.20.1](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-11/PyPI/onnxruntime-gpu/overview/1.20.1) 下载
>
> 还需要特别注意，安装错误的 onnxruntime-gpu 就算不支持GPU，也不会**报错**，而是使用CPU推理。因此，需要测试确认 onnxruntime-gpu 是否安装正确。
>
> 根据官方文档说明 1.18.1 版本同时支持cuda11和cuda12，但没有测试过。如果是cuda11,但安装失败，可以尝试安装1.18.1

### 模型下载

需要下载下面表格中一对文字识别(`rec`)和检测(`det`)模型。

| 模型           | 模型大小      | 模型原始仓库                                                 | 百度网盘下载                                                 | modelscope下载（高速）                                       | 个人评价 |
| -------------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| base_seglink++ | 73.2MB+78MB   | rec[地址](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)，det [地址](https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-line-level_damo/summary) | [地址](https://pan.baidu.com/s/1Vch_5kcL_FqQet5G9pfEJQ?pwd=tjp9) | [v2地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx-v2) | 9分      |
| large          | 73.2MB+46.4MB | rec[地址](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)，det [地址](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary) | [地址](https://pan.baidu.com/s/1Vch_5kcL_FqQet5G9pfEJQ?pwd=tjp9) | [v2地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx-v2) | 8分      |
| small          | 7.4MB+5.2MB   | rec[地址](https://modelscope.cn/models/iic/cv_LightweightEdge_ocr-recognitoin-general_damo/summary)，det [地址](https://www.modelscope.cn/models/iic/cv_proxylessnas_ocr-detection-db-line-level_damo/summary) | [地址](https://pan.baidu.com/s/1Vch_5kcL_FqQet5G9pfEJQ?pwd=tjp9) | [v2地址](https://modelscope.cn/models/mscoder/duguang-ocr-onnx-v2) | 5分      |

>  不同的rec和det可以自由组合使用

> [!WARNING]
> 模型已经更新，旧版模型**不适合**目前项目代码，请下载上表最新版模型文件

### 快速开始

**1 克隆项目**：

```
git clone https://github.com/MGzhou/duguang-ocr-onnx.git
cd duguang-ocr-onnx
```

或手动下载本项目代码

**2 运行示例脚本**：

**base_seglink++**：`demo_seglink.py`

```python
from dgocr.dgocr import DGOCR

# 模型参数
rec_path = r"models\base_seglink++\recognition_model_general"                     # 文字识别模型路径
det_path = r"models\base_seglink++\detection_model_general\model_512x512.onnx"    # 文本检测模型文件路径
img_size=512            # 文本检测模型内部预处理时使用的固定尺寸（单位：像素），与输入图片的实际尺寸无关
model_type = "seglink"  # 模型类型
cpu_thread_num=4        # onnx 运行线程数, 线程越多，识别速度越快
device = "cpu"          # 如果想使用gpu设置为 `device = "gpu"`，同时cpu_thread_num会失效

# 初始化模型
ocr = DGOCR(rec_path, det_path, img_size=img_size, model_type=model_type, device=device, cpu_thread_num=cpu_thread_num)

img1 = "data/det-1.jpg"   	# 图片
img2 = "data/det-2.jpg"
batch_image = [img1, img2]	# 批量，输入的图片数量就是批次大小

# 识别图片
ocr_result = ocr.run(images=batch_image)

# 打印结果
for i in range(len(ocr_result)):
    print(f"第{i+1}张图片结果")
    print(f"{ocr_result[i]}")

# 可视化
for i in range(len(ocr_result)):
    org_path = f"data/det-{i+1}.jpg"
    save_path = f"data/result-det-{i+1}-seg.png"
    ocr.draw(org_path, ocr_result[i], save_path)
    print(f"已经将可视化结果保存至：{save_path}")
```

**large，small**：`demo.py`

```python
from dgocr.dgocr import DGOCR

# 模型参数
rec_path = r"models\large\recognition_model_general"                     # 文字识别模型路径
det_path = r"models\large\detection_model_general\model_512x512.onnx"    # 文本检测模型文件路径
img_size=512            # 文本检测模型内部预处理时使用的固定尺寸（单位：像素），与输入图片的实际尺寸无关
model_type = "common"   # 模型类型
cpu_thread_num=4        # onnx 运行线程数, 线程越多，识别速度越快
device = "cpu"          # 如果想使用gpu设置为 `device = "gpu"`，同时cpu_thread_num会失效

# 初始化模型
ocr = DGOCR(
    rec_path=rec_path, 
    det_path=det_path, 
    img_size=img_size, 
    model_type=model_type, 
    device=device, 
    rec_batch_size=None,
    cpu_thread_num=cpu_thread_num
)


img1 = "data/det-1.jpg"     # 图片
img2 = "data/det-2.jpg"
images = [img1, img2]  # 批量，输入的图片数量就是批次大小
# images = [img1]		# 单张图片
# images = img1			# 单张图片，输出和 [img1] 一样，是按批量时输出

# 识别图片
ocr_result = ocr.run(images=images)

# 打印结果
for i in range(len(ocr_result)):
    print(f"第{i+1}张图片结果")
    print(f"{ocr_result[i]}")

# 可视化
for i in range(len(ocr_result)):
    org_path = f"data/det-{i+1}.jpg"
    save_path = f"data/result-det-{i+1}.png"
    ocr.draw(org_path, ocr_result[i], save_path)
    print(f"已经将可视化结果保存至：{save_path}")
```

> **参数：**
>
> - `img_size` 参数是文本检测模型内部预处理时使用的固定尺寸（单位：像素）。也就是在实际使用时，输入图片的尺寸可以任意的。
> - `device` ，使用GPU推理时，设置为 `gpu`，使用GPU时，推荐显存在4GB以上
> - `rec_batch_size`, 文字识别模型批次大小，默认和文字检测模型一样。而文字检测模型的批次大小是输入图片的数量



> **注意：**
>
> - 输入 `images`, 可以是以列表批量输入处理，也可以单张图片输入处理，例如 `images="data/det-1.jpg"`。 需要注意list批量时，输入的图片数量就是批次大小，如果批次太大，会导致显存不够、cpu运算批次过慢等情况
>
>   images输入还可以是opencv-python读取的图片向量数据，如 `images=[cv2.imread(img1), cv2.imread(img2)]`

**OCR结果说明**

```python
[[[73.0, 612.0], [729.0, 626.0], [728.0, 670.0], [72.0, 656.0]], ('家记忆研究院国家记忆研究院波', 0.9971)]

三部分分别是
[box, (text,score)]; box 为文本框四个点坐标, text 为识别的文本, score 为文本的置信度
```

## 📍GPU 测试

测试中。。。

## 📍CPU 测试

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

## 协议

本项目开源协议是 Apache License 2.0 ；但不包括[AlibabaPuHuiTi-3-45-Light.ttf](https://www.alibabafonts.com/#/font)字体，该字体版权归属[阿里巴巴](https://www.alibabafonts.com/#/font)。

## 感谢

[读光-文字识别-行识别模型-中英-通用领域](https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo/summary)

[读光-文字检测-DBNet行检测模型-中英-通用领域](https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[modelscope](https://github.com/modelscope/modelscope)

[ONNXRuntime CPU推理优化 ](https://rapidai.github.io/RapidOCRDocs/blog/2022/09/23/onnxruntime-cpu%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96/)
