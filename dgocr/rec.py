#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import onnxruntime as rt
import cv2

class DGOCRRecognition:
    def __init__(self, model_path, cpu_thread_num=2) -> None:
        """读光OCR文字识别模型 onnx 版本使用

        Args:
            model (str): 模型路径
            cpu_num_thread (int, optional): CPU线程数, 默认为 2
        """
        self.model_file = self.find_model_file(model_path)
        self.cpu_thread_num = cpu_thread_num
        # 加载模型
        self.load_model()

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
        # rtconfig.enable_profiling = True

        self.sess = rt.InferenceSession(self.model_file["model"], sess_options=rtconfig)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name= self.sess.get_outputs()[0].name
        
        self.labelMapping = dict()
        with open(self.model_file["vocab"], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 2
            for line in lines:
                line = line.strip('\n')
                self.labelMapping[cnt] = line
                cnt += 1
    
    def run(self, image):
        """运行模型"""
        # 读取图片
        if isinstance(image, str):
            image = cv2.imread(image)
        input_data = self.preprocess(image)
        
        # 运行模型
        res = self.sess.run([self.output_name], {self.input_name: input_data})
        outprobs = np.exp(res[0]) / np.sum(np.exp(res[0]), axis=-1, keepdims=True)
        preds = np.argmax(outprobs, -1)
        max_scores = np.amax(outprobs, axis=-1)    # 每个字符的预测概率
        
        # 解码结果
        batchSize, length = preds.shape
        final_str_list = []
        str_score_list = []
        for i in range(batchSize):
            pred_idx = preds[i].data.tolist()
            probability_list = max_scores[i]  # 概率
            last_p = 0
            str_pred = []
            str_score = 1.0
            for j, p in enumerate(pred_idx):
                if p != last_p and p != 0:
                    str_pred.append(self.labelMapping[p])
                    str_score *= probability_list[j]
                last_p = p
            final_str = ''.join(str_pred)
            final_str_list.append(final_str)
            str_score_list.append(str_score)


        return final_str_list, str_score_list

    
    def preprocess(self, img):
        """
        Introduction:
            预处理
        Params:
            img: 图片
        """
        img = self.keepratio_resize(img)
        
        img = np.float32(img)
        chunk_img = []
        for i in range(3):
            left = (300 - 48) * i
            chunk_img.append(img[:, left:left + 300, :])
        
        merge_img = np.concatenate(chunk_img, axis=0)
        data = merge_img.reshape(3, 32, 300, 3) / 255.
        input_data = np.transpose(data, (0, 3, 1, 2))  # .cuda() 
        return input_data

    
    
    def find_model_file(self, model_path):
        """
        Introduction:
            发现模型文件, model.onnx, vocab.txt
        Params:
            model_path: 放置模型的文件夹
        """
        model_file = {"model":"", "vocab":""}
        file_names = os.listdir(model_path)
        if "model.onnx" in file_names:
            model_file["model"] = os.path.join(model_path, "model.onnx")
        else:
            raise ValueError("not find file for model.onnx")
        
        if "vocab.txt" in file_names:
            model_file["vocab"] = os.path.join(model_path, "vocab.txt")
        else:
            raise ValueError("not find file for vocab.txt")
        return model_file
    
    def keepratio_resize(self, img):
        """
        保持宽高比进行缩放

        参数:
            img (numpy.ndarray): 输入图像，形状为 (H, W, C)

        返回:
            numpy.ndarray: 缩放后的图像，形状为 (32, 804, C)
        """
        # 计算当前图像的宽高比
        cur_ratio = img.shape[1] / float(img.shape[0])
        # 定义目标图像的高度和宽度
        mask_height = 32
        mask_width = 804

        # 如果当前图像的宽高比大于目标宽高比，则将图像高度缩放到 32，宽度按比例缩放
        if cur_ratio > float(mask_width) / mask_height:
            cur_target_height = mask_height
            cur_target_width = mask_width
        # 如果当前图像的宽高比小于或等于目标宽高比，则将图像宽度缩放到 804，高度按比例缩放
        else:
            cur_target_height = mask_height
            cur_target_width = int(mask_height * cur_ratio)

        # 使用 OpenCV 的 resize 函数将图像缩放到目标尺寸
        img = cv2.resize(img, (cur_target_width, cur_target_height))

        # 创建一个与目标图像大小相同的全零掩码
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        # 将缩放后的图像复制到掩码的左上角
        mask[:img.shape[0], :img.shape[1], :] = img

        # 将掩码赋值给 img，即返回缩放后的图像
        img = mask

        return img
    

if __name__=="__main__":
    model_path = r"rec_model"

    rec = DGOCRRecognition(model_path)
    img_path = r"img-1.png"
    a = rec.run(img_path)
    print(a)





