#!/usr/bin/env python
# -*- coding:utf-8 -*-

from dgocr.dgocr import DGOCR

# 文字识别模型路径
rec_path = r"models/large/recognition_model_general"
# 文本框检测模型文件路径
det_path = r"models/large/detection_model_general/model_1600x1600.onnx"
img_size=1600  # 模型输入图片大小, 看文本框检测模型的输入图片大小

# 初始化模型
ocr = DGOCR(rec_path, det_path, img_size)

img_path = "data/test.png"

# 识别图片
ocr_result = ocr.run(img_path)

# 打印结果
for i in range(len(ocr_result)):
    print(f"第{i}个框")
    print(f"{ocr_result[i]}")

# 可视化
save_path = "data/result.png"
ocr.draw(img_path, ocr_result, save_path)

