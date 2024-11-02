#!/usr/bin/env python
# -*- coding:utf-8 -*-

from dgocr.dgocr import DGOCR

# 模型参数
rec_path = r"models/large/recognition_model_general"  # 文字识别模型路径
det_path = r"models\base_seglink++\detection_model_general\model_1024x1024.onnx"  # 文本框检测模型文件路径
img_size=1024           # 文本框检测模型的输入图片大小限制
model_type = "seglink"  # 模型类型
cpu_thread_num=2        # onnx 运行线程数
# 初始化模型
ocr = DGOCR(rec_path, det_path, img_size=img_size, model_type=model_type, cpu_thread_num=cpu_thread_num)

img_path = "data/test.png"   # 图片

# 识别图片
ocr_result = ocr.run(img_path)

# 打印结果
for i in range(len(ocr_result)):
    print(f"第{i}个框")
    print(f"{ocr_result[i]}")

# 可视化
save_path = "data/result3.png"
ocr.draw(img_path, ocr_result, save_path)
