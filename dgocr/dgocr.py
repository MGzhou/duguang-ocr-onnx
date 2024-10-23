#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/10/10 09:26:06
@Desc :读光 ocr
'''
import sys
sys.path.append(".")
import cv2
import numpy as np
from PIL import Image

from .rec import DGOCRRecognition
from .det import DGOCRDetection
from .visual import draw_ocr_box_txt
from .utils import crop_image, order_point, preprocess, postprocess

class DGOCR:
    def __init__(self, rec_path, det_path, img_size=1600) -> None:
        """
        初始化模型

        Args:
            rec_path (str): 文字识别模型文件夹路径
            det_path (str): 文本框检测模型文件路径
            img_size (int, optional): 模型限定的图像大小. Defaults to 1600.
        """
        self.rec_path = rec_path
        self.det_path = det_path
        self.img_size = img_size
        self.load_model()

    def load_model(self):
        # 加载模型
        # 文字识别模型
        self.rec_model = DGOCRRecognition(self.rec_path)
        # 文本框检测模型
        self.det_model = DGOCRDetection(self.det_path, self.img_size)

    def run(self, image):
        """
        运行模型

        Args:
            image (str): 图像路径

        Returns:
            ocr_result: 识别结果, [[box, score, text],...]; box 为文本框四个点坐标, score文本框的置信度, text 为识别文本
        """
        # 加载模型
        original_image = cv2.imread(image)
        original_image_size = original_image.shape[:2]

        # 图片预处理
        image_full = preprocess(original_image, (self.img_size, self.img_size))
        current_image_size = image_full.shape[:2]

        # 文本框检测
        det_result = self.det_model.run(image_full)
        boxes,scores  = np.array(det_result['polygons']), det_result['scores']

        pos_list = []
        text_list = []
        for i in range(boxes.shape[0]):
            pts = order_point(boxes[i])
            # print(f"原始框:{boxes[i]},   变换后：{pts}")
            image_crop = crop_image(image_full, pts)  # 裁剪文本框
            # 保存裁剪的图片
            # path2 = rf"image_crop/{i}.png"
            # cv2.imwrite(path2, image_crop)
            result = self.rec_model.run(image_crop)   # 文字识别

            if len(result) > 0:
                pos_list.append(pts.tolist())
                text_list.append(result[0])
        # 后处理
        pos_list = postprocess(original_image_size, current_image_size, pos_list)
        ocr_result = []
        for i in range(len(pos_list)):
            if len(pos_list[i]) > 0:
                ocr_result.append([pos_list[i], scores[i], text_list[i]])

        return ocr_result

    def draw(self, img_path, ocr_result, save_path):
        """
        绘制识别结果

        Args:
            image (str): 图像路径
            ocr_result (list): 识别结果
            save_path (str): 保存路径
        """
        image = Image.open(img_path).convert('RGB')
        # 从orc_result获取 boxes, scores, text
        boxs = [i[0] for i in ocr_result]
        texts = [i[2] for i in ocr_result]
        image = draw_ocr_box_txt(image, boxs, texts)
        im_show = Image.fromarray(image)
        im_show.save(save_path)



if __name__=="__main__":
    rec_path = r""
    det_path = r""
    img_size=1600

    img_path = r""

    # 初始化模型
    ocr = DGOCR(rec_path, det_path, img_size)

    ocr_result = ocr.run(img_path)

    for i in range(len(ocr_result)):
        print(f"第{i}个框")
        print(f"{ocr_result[i]}")






























