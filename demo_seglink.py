#!/usr/bin/env python
# -*- coding:utf-8 -*-

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

img1 = "data/det-1.jpg"     # 图片
img2 = "data/det-2.jpg"
batch_image = [img1, img2]  # 批量，输入的图片数量就是批次大小

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

