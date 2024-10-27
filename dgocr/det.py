#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import onnxruntime as rt

class DGOCRDetection:
    def __init__(self, model_path, img_size=1600, cpu_num_thread=2):
        """读光OCR文字框检测模型 onnx 版本使用

        Args:
            model_path (str): 模型路径, xxx.onnx
            img_size (int, optional): 模型限制图片大小, 默认为 1600, 要是800的倍数增加，越大越精确，但速度会变慢
            cpu_num_thread (int, optional): CPU线程数, 默认为 2
        """
        self.model_path = model_path
        self.img_size = img_size
        self.cpu_num_thread = cpu_num_thread
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        # 创建一个SessionOptions对象
        rtconfig = rt.SessionOptions()
        
        # 设置CPU线程数
        rtconfig.intra_op_num_threads = self.cpu_num_thread
        # 并行 ORT_PARALLEL  顺序 ORT_SEQUENTIAL
        rtconfig.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        rtconfig.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        rtconfig.log_severity_level = 4
        rtconfig.enable_cpu_mem_arena = False
        # rtconfig.enable_profiling = True  #  生成一个类似onnxruntime_profile__2023-05-07_09-02-15.json的日志文件，包含详细的性能数据（线程、每个运算符的延迟等）。

        self.ort_session = rt.InferenceSession(self.model_path, sess_options=rtconfig)

        
    def run(self, image):
        """运行模型"""""
        if isinstance(image, str):
            image = cv2.imread(image)
        height, width, _ = image.shape
        image_resize = cv2.resize(image, (self.img_size,self.img_size))        
        image_resize = image_resize - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image_resize /= 255.
        image_resize = np.expand_dims(image_resize.transpose(2, 0, 1), axis=0)

        outputs = self.ort_session.run(['pred'], {'images': image_resize})

        ## 后处理
        thresh = 0.2
        pred = outputs[0]
        segmentation = pred > thresh
        boxes, scores = boxes_from_bitmap(pred, segmentation, width,
                                            height, is_numpy=True) 

        return {"polygons":boxes, "scores":scores}



    def export_onnx(self, model_path, output_path, img_size=1600):
        """
        该模型 onnx 可能与硬件有关，如果报错，尝试自行导出onnx模型
        导出模型, modelscope 版本 1.9.5

        Args:
            model_path (str): 模型路径
            output_path (str): 输出路径, 会在输出路径下生成 model.onnx 模型文件
            img_size (int, optional): 图片大小, 默认为 1600, 要是800的倍数增加，越大越精确，但速度会变慢

        """
        from modelscope.models import Model
        from modelscope.exporters import Exporter
        model = Model.from_pretrained(model_path)
        Exporter.from_model(model).export_onnx(
            input_shape=(1,3,img_size,img_size), output_dir=output_path)  # input_shape 的图片尺寸要安装 800倍数增加


"""
解析工具函数
"""

def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height, is_numpy=False):
    """
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    """
    if is_numpy:
        bitmap = _bitmap[0]
        pred = pred[0]
    else:
        bitmap = _bitmap.cpu().numpy()[0]
        pred = pred.cpu().detach().numpy()[0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:1000]:
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)

        score = box_score_fast(pred, points.reshape(-1, 2))
        if 0.3 > score:
            continue

        box = unclip(points, unclip_ratio=1.5).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)

        if sside < 3 + 2:
            continue

        box = np.array(box).astype(np.int32)
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.reshape(-1).tolist())
        scores.append(score)
    return boxes, scores


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


if __name__ == '__main__':
    
    model_path = "model_1600x1600.onnx"
    image_path = "img-1.png"
    det = DGOCRDetection(model_path)
    
    t1 = time.time()
    
    boxes = det.run(image_path)

    print(boxes)
    t2 = time.time()
    print(f"time: {t2-t1}")









