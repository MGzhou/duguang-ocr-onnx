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
from .det_seglink import SegLinkOCRDetection
from .visual import draw_ocr_box_txt
from .utils import crop_image, order_point, preprocess, postprocess

class DGOCR:
    def __init__(
            self, 
            rec_path, 
            det_path, 
            img_size=1600, 
            device="cpu",
            rec_batch_size=None,
            cpu_thread_num=2, 
            model_type="common"
        ) -> None:
        """
        初始化模型

        Args:
            rec_path (str): 文字识别模型文件夹路径
            det_path (str): 文本框检测模型文件路径
            img_size (int): 模型限定的图像大小. Defaults to 1600.
            device (str): 运行设备，默认是使用cpu, 可以设置为 gpu 以使用显卡加速
            rec_batch_size: 文字识别批次大小，默认和输入的图片数量一样
            cpu_thread_num (int): CPU线程数, 默认为 2. 越大速度越快，但占用资源越多。如果device设置为gpu则不生效。
            model_type (str): 模型类型, 默认为 "common" 生成模型, 可选 "seglink" 推理模型.
        """
        self.rec_path = rec_path
        self.det_path = det_path
        self.img_size = img_size
        self.device = device
        self.rec_batch_size = rec_batch_size
        self.cpu_thread_num = cpu_thread_num
        self.model_type = model_type

        self.load_model()

    def load_model(self):
        # 加载模型
        # 文字识别模型
        self.rec_model = DGOCRRecognition(self.rec_path, self.device, self.cpu_thread_num)
        # 文本框检测模型
        if self.model_type == "seglink":
            self.det_model = SegLinkOCRDetection(self.det_path, self.img_size, self.device, self.cpu_thread_num)
        else:
            self.det_model = DGOCRDetection(self.det_path, self.img_size, self.device, self.cpu_thread_num)

    def run(self, images):
        """
        运行模型

        Args:
            images (str): 图像路径列表, 或者cv读取的图片向量

        Returns:
            ocr_result: 识别结果, [[box, score, text],...]; box 为文本框四个点坐标, score文本框的置信度, text 为识别文本
        """
        # 加载模型
        image_datas, org_img_sizes = self.preprocess(images)

        batch_size = len(image_datas)
        if self.rec_batch_size is None:
            rec_batch_size = batch_size
        else:
            rec_batch_size = self.rec_batch_size
        
        # 文本检测
        det_result = self.det_model.run(image_datas)

        batch_boxes  = det_result['polygons']

        image_chunks, chunk_boxes, image_chunk_start_ids = self.crop_image_process(image_datas, batch_boxes)

        rec_texts = []
        rec_probs = []
        for i in range(0, len(image_chunks), rec_batch_size):
            # 文本识别
            rec_result = self.rec_model.run(image_chunks[i:i+rec_batch_size])
            rec_texts.extend(rec_result["preds"])
            rec_probs.extend(rec_result["probs"])
        
        # 后处理
        batch_ocr_result = []
        image_chunk_start_ids.append(len(image_chunks))
        for i in range(len(image_chunk_start_ids)-1):
            one_image_result = []

            start = image_chunk_start_ids[i]
            end = image_chunk_start_ids[i+1]

            for j in range(start, end):
                text = rec_texts[j]
                if len(text.strip()) == 0:
                    continue
                one_image_result.append([chunk_boxes[j], (rec_texts[j], rec_probs[j])])
            batch_ocr_result.append(one_image_result)

        return batch_ocr_result
    
    def preprocess(self, inputs):
        """预处理"""
        image_datas = []
        org_img_sizes = []
        if not isinstance(inputs, list):
            if isinstance(inputs, str):
                image_datas = [cv2.imread(inputs)]
            else:
                image_datas = [inputs]
        else:
            image_datas = [cv2.imread(img) if isinstance(img, str) else img for img in inputs]
        org_img_sizes = [img.shape[:2] for img in image_datas]
        return image_datas, org_img_sizes

    def crop_image_process(self, image_datas, boxs_list):

        image_chunks = []
        image_chunk_start_ids = []
        chunk_boxes = []
        for i, image in enumerate(image_datas):
            image_chunk_start_ids.append(len(image_chunks))
            boxes = np.array(boxs_list[i])
            for j in range(boxes.shape[0]):
                pts = order_point(boxes[j])
                image_chunks.append(crop_image(image, pts))  # 裁剪文本框
                chunk_boxes.append(pts.tolist())
        return image_chunks, chunk_boxes, image_chunk_start_ids

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
        texts = [i[1][0] for i in ocr_result]
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






























