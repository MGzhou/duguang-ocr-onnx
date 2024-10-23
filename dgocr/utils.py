#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import cv2

def crop_image(img, position):
    """裁剪图像脚本"""
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


def order_point(coor):
    """
    作用是对一个四边形的四个顶点坐标进行排序，使得排序后的顶点按照顺时针或逆时针方向围绕中心点排列。
    Example:
        >>> order_point([1187  879 1320  879 1320  905 1187  905])
        array([ [1187.,  879.]
                [1320.,  879.]
                [1320.,  905.]
                [1187.,  905.]])
    """
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points



def preprocess(image, target_size=(1600, 1600), fill_color=(255, 255, 255)):  
    """
    图片预处理，缩放和填充图片
    例如 (800x400)  --放大--> (1600x800) --填充--> (1600x1600)
    Args:
        image (numpy.ndarray): 输入图像，形状为 (H, W, C), 或者图片路径
        target_size (tuple): 目标尺寸，默认为 (1600, 1600)
        fill_color (tuple): 填充颜色，默认为 (255, 255, 255)
    Returns:
        numpy.ndarray: 预处理后的图像
    """
    # 读取图片
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:  
        raise ValueError(f"Image not found at path: {image}")  
    
    # 获取原始图片尺寸  
    original_height, original_width = image.shape[:2]  
    
    # 计算缩放比例
    aspect_ratio = original_width / original_height  
    if aspect_ratio > 1:  
        # 图片更宽  
        new_width = target_size[0]  
        new_height = int(target_size[0] / aspect_ratio)  
    else:  
        # 图片更高或等宽高  
        new_height = target_size[1]  
        new_width = int(target_size[1] * aspect_ratio)  
    
    # 缩放图片  
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)  
    
    # 计算填充区域大小  
    pad_width = (target_size[0] - new_width) // 2  
    pad_height = (target_size[1] - new_height) // 2  
    
    # 创建填充后的图像
    padded_image = np.full(target_size + (3,), fill_color, dtype=np.uint8)  
    padded_image[pad_height:pad_height + new_height, pad_width:pad_width + new_width] = resized_image
    return padded_image

def postprocess(original_image_size, current_image_size, pos_list):
    """
    后处理，将图片坐标还原为原始图片尺寸
    """
    o_h, o_w = original_image_size
    c_h, c_w = current_image_size

    # 最大的边
    max_side = 0    #  0:高，1:宽
    if o_h < o_w:
        max_side = 1

    o_max = max(o_h, o_w)
    o_min = min(o_h, o_w)
    c_max = max(c_h, c_w)
    scale = c_max / o_max
    padding_size = (c_max - (o_min * scale)) // 2

    # (x,y)  x :宽 y:高
    # 还原坐标
    for i in range(len(pos_list)):
        for j in range(len(pos_list[i])):
            if max_side == 0:  # 高, 填充宽度, 宽 - padding_size

                pos_list[i][j][0] = int((pos_list[i][j][0]-padding_size) / scale)
                
                pos_list[i][j][1] = int(pos_list[i][j][1] / scale)

                if pos_list[i][j][0] > o_w:
                    pos_list[i][j][0] = o_w
                elif pos_list[i][j][0] < 0:
                    pos_list[i][j][0] = 0

                if pos_list[i][j][1] > o_h:
                    pos_list[i][j][1] = o_h
                elif pos_list[i][j][1] < 0:
                    pos_list[i][j][1] = 0
            else:            # 宽，填充高度，高 - padding_size
                pos_list[i][j][0] = int(pos_list[i][j][0] / scale)
                pos_list[i][j][1] = int((pos_list[i][j][1]-padding_size) / scale)

                if pos_list[i][j][0] > o_w:
                    pos_list[i][j][0] = o_w
                elif pos_list[i][j][0] < 0:
                    pos_list[i][j][0] = 0

                if pos_list[i][j][1] > o_h:
                    pos_list[i][j][1] = o_h
                elif pos_list[i][j][1] < 0:
                    pos_list[i][j][1] = 0
    
    # 计算面积，如果面积为0，则赋值[]
    for i in range(len(pos_list)):
        if calculate_polygon_area(pos_list[i]) <= 0:
            pos_list[i] = []
    return pos_list

def calculate_polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2