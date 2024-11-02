#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/10/31 16:10:37
@Desc :None
'''


import cv2
import numpy as np
import onnxruntime as rt
from .utils_seglink import decode_segments_links_python, combine_segments_python
from .utils_seglink import cal_width, nms_python, rboxes_to_polygons

class SegLinkOCRDetection():
    """ 基于SegLink的读光OCR文本检测模型
    """
    def __init__(self, model: str, cpu_thread_num: int=2):
        """
        Args:
            model (str): onnx model path
        """
        self.model_path = model
        self.cpu_thread_num = cpu_thread_num
        self.load_model()
        self.output = {}
    
    def load_model(self):
        """加载模型"""
        # 创建一个SessionOptions对象
        rtconfig = rt.SessionOptions()
        
        # 设置CPU线程数
        rtconfig.intra_op_num_threads = self.cpu_thread_num
        # 并行 ORT_PARALLEL  顺序 ORT_SEQUENTIAL
        rtconfig.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        rtconfig.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        rtconfig.log_severity_level = 4
        rtconfig.enable_cpu_mem_arena = False
        # 加载模型
        self.sess = rt.InferenceSession(self.model_path, sess_options=rtconfig)
    
    def run(self, input):
        """
        文本检测
        Args:
            input (`Image`): 图片路径
        """
        # 进行文本检测
        out = self.preprocess(input)
        out = self.forward(out)
        out = self.postprocess(out)
        return out
    
    def forward(self, input):
        """前向传播"""
        # 图片数据
        input_images = input['img']
        # onnx预测
        input_name = self.sess.get_inputs()[0].name
        output_names = ['dete_0/conv_cls/BiasAdd:0', 'dete_0/conv_lnk/BiasAdd:0', 'dete_0/conv_reg/BiasAdd:0', 
                        'dete_1/conv_cls/BiasAdd:0', 'dete_1/conv_lnk/BiasAdd:0', 'dete_1/conv_reg/BiasAdd:0', 
                        'dete_2/conv_cls/BiasAdd:0', 'dete_2/conv_lnk/BiasAdd:0', 'dete_2/conv_reg/BiasAdd:0', 
                        'dete_3/conv_cls/BiasAdd:0', 'dete_3/conv_lnk/BiasAdd:0', 'dete_3/conv_reg/BiasAdd:0', 
                        'dete_4/conv_cls/BiasAdd:0', 'dete_4/conv_lnk/BiasAdd:0', 'dete_4/conv_reg/BiasAdd:0', 
                        'dete_5/conv_cls/BiasAdd:0', 'dete_5/conv_lnk/BiasAdd:0', 'dete_5/conv_reg/BiasAdd:0']

        temp_all_maps = self.sess.run([output_name for output_name in output_names], {input_name: input_images})
        all_maps = [temp_all_maps[i:i + 3] for i in range(0, len(temp_all_maps), 3)]

        # 模型推理结果解码
        all_nodes, all_links, all_reg = [], [], []
        for i, maps in enumerate(all_maps):
            cls_maps, lnk_maps, reg_maps = maps[0], maps[1], maps[2]
            offset_variance = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            reg_maps = np.multiply(reg_maps, offset_variance)

            # 将softmax应用到每个类映射
            cls_maps_reshaped = cls_maps.reshape(-1, 2)
            cls_prob = np.exp(cls_maps_reshaped) / np.sum(np.exp(cls_maps_reshaped), axis=1, keepdims=True)

            # 计算链接概率
            lnk_maps_reshaped = lnk_maps.reshape(-1, 4)
            lnk_prob_pos = np.exp(lnk_maps_reshaped[:, :2]) / np.sum(np.exp(lnk_maps_reshaped[:, :2]), axis=1, keepdims=True)
            lnk_prob_mut = np.exp(lnk_maps_reshaped[:, 2:]) / np.sum(np.exp(lnk_maps_reshaped[:, 2:]), axis=1, keepdims=True)
            lnk_prob = np.concatenate([lnk_prob_pos, lnk_prob_mut], axis=1)

            all_nodes.append(cls_prob)
            all_links.append(lnk_prob)
            all_reg.append(reg_maps)

        # decode segments and links
        image_size = np.array(list(input_images.shape[1:3]))
        segments, group_indices, segment_counts, _ = decode_segments_links_python(
            image_size,
            all_nodes,
            all_links,
            all_reg,
            anchor_sizes=[6., 11.84210526, 23.68421053, 45., 90., 150.]
        )
        # combine segments
        combined_rboxes, combined_counts = combine_segments_python(segments, group_indices, segment_counts)
        
        self.output['combined_rboxes'] = combined_rboxes
        self.output['combined_counts'] = combined_counts
        return self.output

    def preprocess(self, input):
        """图片预处理"""
        # pillow
        # img = Image.open(input)  
        # img = ImageOps.exif_transpose(img)
        # img = img.convert("RGB")
        # img = np.array(img)
        # cv2
        if isinstance(input, str):
            img = cv2.imread(input)
        else:
            img = input
        # 将图像从 BGR 转换为 RGB（OpenCV 默认读取的是 BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        img_pad = np.zeros((max(h, w), max(h, w), 3), dtype=np.float32)
        img_pad[:h, :w, :] = img

        resize_size = 1024
        img_pad_resize = cv2.resize(img_pad, (resize_size, resize_size))
        img_pad_resize = cv2.cvtColor(img_pad_resize, cv2.COLOR_RGB2BGR)
        img_pad_resize = img_pad_resize - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        resize_size = np.array([resize_size, resize_size])
        orig_size = np.array([max(h, w), max(h, w)])
        self.output['orig_size'] = orig_size
        self.output['resize_size'] = resize_size

        result = {'img': np.expand_dims(img_pad_resize, axis=0)}
        return result
    
    def postprocess(self, inputs):
        """图片后处理"""
        rboxes = inputs['combined_rboxes'][0]
        count = inputs['combined_counts'][0]
        if count == 0 or count < rboxes.shape[0]:
            # raise Exception('No text detected')
            return {"polygons": []}
        rboxes = rboxes[:count, :]

        # convert rboxes to polygons and find its coordinates on the original image
        orig_h, orig_w = inputs['orig_size']
        resize_h, resize_w = inputs['resize_size']
        polygons = rboxes_to_polygons(rboxes)
        scale_y = float(orig_h) / float(resize_h)
        scale_x = float(orig_w) / float(resize_w)

        # confine polygons inside image
        polygons[:, ::2] = np.maximum(0, np.minimum(polygons[:, ::2] * scale_x, orig_w - 1))
        polygons[:, 1::2] = np.maximum(0, np.minimum(polygons[:, 1::2] * scale_y, orig_h - 1))
        polygons = np.round(polygons).astype(np.int32)

        # nms
        dt_n9 = [o + [cal_width(o)] for o in polygons.tolist()]
        dt_nms = nms_python(dt_n9)
        dt_polygons = np.array([o[:8] for o in dt_nms])
        result = {"polygons": dt_polygons}
        return result



if __name__=="__main__":
    import time
    img1 = r'1.png'
    a = SegLinkOCRDetection(model=r"E:\GithubCode\my-open-project\duguang-ocr-onnx\models\base_seglink++\detection_model_general\model_1024x1024.onnx")
    # for _ in range(10):
    t1 = time.time()
    result = a.run(img1)
    print(result)
    print(f"耗时 = {time.time() - t1}")

