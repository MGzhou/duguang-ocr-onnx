#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import onnxruntime as rt

class DGOCRDetection:
    def __init__(
        self, 
        model_path, 
        img_size: int =1600, 
        device: str ="cpu",
        cpu_thread_num: int =2
    ):
        """读光OCR文字框检测模型 onnx 版本使用

        Args:
            model_path (str): 模型路径, xxx.onnx
            img_size (int, optional): 模型限制图片大小, 默认为 1600, 越大越精确，但速度会变慢
            device (str): 运行设备，默认是使用cpu, 可以设置为 gpu 以使用显卡加速
            cpu_thread_num (int, optional): CPU线程数, 默认为 2
        """
        self.model_path = model_path
        self.img_size = img_size
        self.device = device
        self.cpu_thread_num = cpu_thread_num
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        if self.device == "gpu":
            providers = ["CUDAExecutionProvider"]
            self.ort_session = rt.InferenceSession(self.model_path, providers=providers)
        else:
            # 创建一个SessionOptions对象
            rtconfig = rt.SessionOptions()
            # 设置CPU线程数
            rtconfig.intra_op_num_threads = self.cpu_thread_num
            # 并行 ORT_PARALLEL  顺序 ORT_SEQUENTIAL
            rtconfig.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            rtconfig.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            rtconfig.log_severity_level = 4
            rtconfig.enable_cpu_mem_arena = False
            # rtconfig.enable_profiling = True  #  生成一个类似onnxruntime_profile__2023-05-07_09-02-15.json的日志文件，包含详细的性能数据（线程、每个运算符的延迟等）。
            providers = ["CPUExecutionProvider"]
            self.ort_session = rt.InferenceSession(self.model_path, sess_options=rtconfig, providers=providers)

        
    def run(self, inputs):
        """
        运行模型
        
        Returns:
            {
                "polygons":[[<img1_box1>, <img1_box2>...], [<img2_box1>,...]...],
                "scores": [[<img1_box1_prob>, <img1_box2_prob>...], [<img2_box1_prob>,...]...]
            }
        """""
        inputs = self.batch_preprocess(inputs)
        orig_size_list = inputs["orig_size"]

        outputs = self.ort_session.run(['pred'], {'images': inputs["img"]})
        
        boxes, scores = self.postprocess(outputs, orig_size_list)
        # 批量结果
        return {"polygons":boxes, "scores":scores}

    def postprocess(self, outputs, orig_size_list):
        thresh = 0.2  # 阈值
        preds = outputs[0]  # [batch, 1, 1600, 1600]
        if preds.shape[0] == 1:
            preds = [preds]
        else:
            preds = np.split(preds, preds.shape[0], axis=0)

        boxes_list, scores_list = [], []
        for i, pred in enumerate(preds):
            
            pred = pred[0]
            segmentation = pred > thresh
            boxes, scores = boxes_from_bitmap(
                pred, 
                segmentation, 
                orig_size_list[i][1],
                orig_size_list[i][0], 
                is_numpy=True
            ) 
            boxes_list.append(boxes)
            scores_list.append(scores)
        return boxes_list, scores_list

    def batch_preprocess(self, inputs):
        if not isinstance(inputs, list):
            image, height, width = self.preprocess(inputs)
            result = {"img":image, "orig_size":[(height, width)]}

        else:
            im_list = []
            im_orig_size_list = []
            for im in inputs:
                temp_image, temp_height, temp_width = self.preprocess(im)
                im_list.append(temp_image)
                im_orig_size_list.append((temp_height,temp_width))

            batch_im = np.vstack(im_list)
            result = {
                "img":batch_im,
                "orig_size": im_orig_size_list,
            }
        return result

    def preprocess(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        height, width, _ = image.shape
        image_resize = cv2.resize(image, (self.img_size, self.img_size))        
        image_resize = image_resize - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image_resize /= 255.
        image_resize = np.expand_dims(image_resize.transpose(2, 0, 1), axis=0)

        return image_resize, height, width


"""
解码工具函数
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
    
    model_path = r"xxx\model_1600x1600.onnx"
    img1 = r'xxx'
    img2 = r'xxx'
    det = DGOCRDetection(model_path, img_size=1600, device="cpu")
    t1 = time.time()
    # for i in range(100):
    boxes = det.run([img1, img2, img1])

    print(boxes)
    print(boxes["polygons"])
    t2 = time.time()
    print(f"time: {t2-t1}")