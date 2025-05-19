

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
    """
    基于SegLink的读光OCR文本检测模型
    """
    def __init__(
        self, 
        model: str,
        img_size: int =1024, 
        device: str = "cpu",
        cpu_thread_num: int=2,
        ):
        """
        Args:
            model (str): onnx model path
            img_size (int, optional): 模型限制图片大小, 默认为 1024, 越大越精确，但速度会变慢
            device (str): 运行设备，默认是使用cpu, 可以设置为 gpu 以使用显卡加速
            cpu_thread_num (int): cpu 线程数量
        """
        self.model_path = model
        self.device = device
        self.img_size = img_size
        self.cpu_thread_num = cpu_thread_num
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        # 创建一个SessionOptions对象
        if self.device == "gpu":
            providers = ["CUDAExecutionProvider"]  # ("CUDAExecutionProvider", {"device_id": 0})
            # 加载模型
            self.sess = rt.InferenceSession(self.model_path, providers=providers)
            
        else:
            rtconfig = rt.SessionOptions()
            # 设置CPU线程数
            rtconfig.intra_op_num_threads = self.cpu_thread_num
            # 并行 ORT_PARALLEL  顺序 ORT_SEQUENTIAL
            rtconfig.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            rtconfig.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            rtconfig.log_severity_level = 4
            rtconfig.enable_cpu_mem_arena = False
            providers = ["CPUExecutionProvider"]
            # 加载模型
            self.sess = rt.InferenceSession(self.model_path, sess_options=rtconfig, providers=providers)
    
    def run(self, inputs):
        """
        文本检测
        Args:
            input (list[str], list[np.ndarray]): 图片路径或图片向量
        Returns:
            {"polygons":[[<img1_box1>, <img1_box2>...], [<img2_box1>,...]...]}
        """
        img_info = {}
        out = self.batch_preprocess(inputs)
        img_info["orig_size"] = out["orig_size"]
        img_info["resize_size"] = out["resize_size"]

        out = self.forward(out["img"])
        img_info["combined_rboxes"] = out["combined_rboxes"]
        img_info["combined_counts"] = out["combined_counts"]
        out = self.postprocess(img_info)
        return out
    
    def forward(self, input_images):
        """前向传播"""
        # onnx预测
        input_name = self.sess.get_inputs()[0].name
        output_names = ['dete_0/conv_cls/BiasAdd:0', 'dete_0/conv_lnk/BiasAdd:0', 'dete_0/conv_reg/BiasAdd:0', 
                        'dete_1/conv_cls/BiasAdd:0', 'dete_1/conv_lnk/BiasAdd:0', 'dete_1/conv_reg/BiasAdd:0', 
                        'dete_2/conv_cls/BiasAdd:0', 'dete_2/conv_lnk/BiasAdd:0', 'dete_2/conv_reg/BiasAdd:0', 
                        'dete_3/conv_cls/BiasAdd:0', 'dete_3/conv_lnk/BiasAdd:0', 'dete_3/conv_reg/BiasAdd:0', 
                        'dete_4/conv_cls/BiasAdd:0', 'dete_4/conv_lnk/BiasAdd:0', 'dete_4/conv_reg/BiasAdd:0', 
                        'dete_5/conv_cls/BiasAdd:0', 'dete_5/conv_lnk/BiasAdd:0', 'dete_5/conv_reg/BiasAdd:0']

        batch_all_maps = self.sess.run([output_name for output_name in output_names], {input_name: input_images})

        # 切分
        batch_size = batch_all_maps[0].shape[0]
        new_batch_all_maps = [[] for _ in range(batch_size)]
        for all_maps in batch_all_maps:
            temp = [all_maps[i:i+1, :, :, :] for i in range(batch_all_maps[0].shape[0])]
            for j in range(batch_size):
                new_batch_all_maps[j].append(temp[j])
        # 解码
        combined_rboxe_list, combined_count_list = [], []

        input_images_shape = input_images.shape[1:3]
        for i in range(batch_size):
            combined_rboxe, combined_count = self.decode_model_output(new_batch_all_maps[i], input_images_shape)
            combined_rboxe_list.append(combined_rboxe)
            combined_count_list.append(combined_count)
        
        return {"combined_rboxes":combined_rboxe_list, "combined_counts":combined_count_list}
        

    def decode_model_output(self, all_maps, input_images_shape):
        all_maps = [all_maps[i:i + 3] for i in range(0, len(all_maps), 3)]

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
        image_size = np.array(list(input_images_shape))  # 
        segments, group_indices, segment_counts, _ = decode_segments_links_python(
            image_size,
            all_nodes,
            all_links,
            all_reg,
            anchor_sizes=[6., 11.84210526, 23.68421053, 45., 90., 150.]
        )
        # combine segments
        combined_rboxe, combined_count = combine_segments_python(segments, group_indices, segment_counts)
        
        return combined_rboxe, combined_count

    def batch_preprocess(self, inputs):
        if not isinstance(inputs, list):
            result = self.preprocess(inputs)
            result["orig_size"] = [result["orig_size"]]
            result["resize_size"] = [result["resize_size"]]
        else:
            im_list = []
            im_orig_size_list = []
            im_resize_size_list = []
            for im in inputs:
                temp = self.preprocess(im)
                im_list.append(temp["img"])
                im_orig_size_list.append(temp["orig_size"])
                im_resize_size_list.append(temp["resize_size"])

            batch_im = np.vstack(im_list)
            result = {
                "img":batch_im,
                "orig_size": im_orig_size_list,
                "resize_size": im_resize_size_list
            }

        return result

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

        resize_size = self.img_size
        img_pad_resize = cv2.resize(img_pad, (resize_size, resize_size))
        img_pad_resize = cv2.cvtColor(img_pad_resize, cv2.COLOR_RGB2BGR)
        img_pad_resize = img_pad_resize - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        resize_size = np.array([resize_size, resize_size])
        orig_size = np.array([max(h, w), max(h, w)])

        result = {
            'img': np.expand_dims(img_pad_resize, axis=0), 
            "orig_size": orig_size,
            "resize_size": resize_size
        }
        return result
    
    def postprocess(self, inputs):
        """图片后处理"""
        res_list = []
        for i in range(len(inputs["orig_size"])):
            rboxes = inputs['combined_rboxes'][i][0]
            count = inputs['combined_counts'][i][0]
            if count == 0 or count < rboxes.shape[0]:
                # raise Exception('No text detected')
                return {"polygons": []}
            rboxes = rboxes[:count, :]

            # convert rboxes to polygons and find its coordinates on the original image
            orig_h, orig_w = inputs['orig_size'][i]
            resize_h, resize_w = inputs['resize_size'][i]
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
            res_list.append(dt_polygons.tolist())
        result = {"polygons": res_list}
        return result



if __name__=="__main__":
    import time
    img1 = r'xxx'
    img2 = r'xxx'

    model=r"xxx/model_1024x1024.onnx"
    a = SegLinkOCRDetection(model=model, device="gpu")
    result = a.run([img1,img2])
    print(result)
    t1 = time.time()
    # for _ in range(10):
    #     result = a.run(img1)
    #     print(result)
    print(f"耗时 = {time.time() - t1}")

