#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/10/31 16:10:37
@Desc :None
'''

import numpy as np

# 常量
OFFSET_DIM = 6
RBOX_DIM = 5

N_LOCAL_LINKS = 8
N_CROSS_LINKS = 4
N_SEG_CLASSES = 2
N_LNK_CLASSES = 4

MATCH_STATUS_POS = 1
MATCH_STATUS_NEG = -1
MATCH_STATUS_IGNORE = 0
MUT_LABEL = 3
POS_LABEL = 1
NEG_LABEL = 0

N_DET_LAYERS = 6

FLAGS_NODE_THRESHOLD = 0.4
FLAGS_LINK_THRESHOLD = 0.6

""" 
decode_segments_links_python 解码
"""

def decode_segments_links_python(image_size, all_nodes, all_links, all_reg, anchor_sizes):
    batch_size = 1  # batch_size为1

    # 将输入的数据展平并拼接
    all_nodes_flat = np.concatenate([np.reshape(o, (batch_size, -1, N_SEG_CLASSES)) for o in all_nodes],axis=1)
    all_links_flat = np.concatenate([np.reshape(o, (batch_size, -1, N_LNK_CLASSES)) for o in all_links],axis=1)
    all_reg_flat = np.concatenate([np.reshape(o, (batch_size, -1, OFFSET_DIM)) for o in all_reg], axis=1)

    # 使用decode_batch函数解码
    segments, group_indices, segment_counts, group_indices_all = decode_batch(
        all_nodes_flat, all_links_flat, all_reg_flat, image_size, np.array(anchor_sizes)
    )
    return segments, group_indices, segment_counts, group_indices_all

def decode_batch(all_nodes, all_links, all_reg, image_size, anchor_sizes):
    batch_size = all_nodes.shape[0]
    batch_segments = []
    batch_group_indices = []
    batch_segments_counts = []
    batch_group_indices_all = []
    for image_id in range(batch_size):
        image_node_scores = all_nodes[image_id, :, :]
        image_link_scores = all_links[image_id, :, :]
        image_reg = all_reg[image_id, :, :]
        image_segments, image_group_indices, image_segments_counts, image_group_indices_all = decode_image(
            image_node_scores, image_link_scores, image_reg, image_size,anchor_sizes)
        batch_segments.append(image_segments)
        batch_group_indices.append(image_group_indices)
        batch_segments_counts.append(image_segments_counts)
        batch_group_indices_all.append(image_group_indices_all)
    max_count = np.max(batch_segments_counts)
    for image_id in range(batch_size):
        if not batch_segments_counts[image_id] == max_count:
            batch_segments_pad = (max_count - batch_segments_counts[image_id]) * [OFFSET_DIM * [0.0]]
            batch_segments[image_id] = np.vstack((batch_segments[image_id], np.array(batch_segments_pad)))
            batch_group_indices[image_id] = np.hstack((batch_group_indices[image_id],
                 np.array((max_count - batch_segments_counts[image_id]) * [-1])))
    a = np.asarray(batch_segments, np.float32)
    b = np.asarray(batch_group_indices,np.int32)
    c = np.asarray(batch_segments_counts,np.int32)
    d = np.asarray(batch_group_indices_all,np.int32)
    return a, b, c, d


def decode_image(image_node_scores, image_link_scores, image_reg, image_size, anchor_sizes):
    map_size = []
    offsets_defaults = []
    offsets_default_node = 0
    offsets_default_link = 0
    
    for i in range(N_DET_LAYERS):
        offsets_defaults.append([offsets_default_node, offsets_default_link])
        
        map_size.append(image_size // (2**(2 + i)))
        offsets_default_node += map_size[i][0] * map_size[i][1]
        if i == 0:
            offsets_default_link += map_size[i][0] * map_size[i][1] * N_LOCAL_LINKS
        else:
            offsets_default_link += map_size[i][0] * map_size[i][1] * (N_LOCAL_LINKS + N_CROSS_LINKS)

    image_group_indices_all = decode_image_by_join(image_node_scores,
                                                   image_link_scores,
                                                   FLAGS_NODE_THRESHOLD,
                                                   FLAGS_LINK_THRESHOLD,
                                                   map_size, offsets_defaults)
    image_group_indices_all -= 1
    image_group_indices = image_group_indices_all[np.where(image_group_indices_all >= 0)[0]]
    image_segments_counts = len(image_group_indices)
    # convert image_reg to segments with scores(OFFSET_DIM+1)
    image_segments = np.zeros((image_segments_counts, OFFSET_DIM), dtype=np.float32)
    for i, offsets in enumerate(np.where(image_group_indices_all >= 0)[0]):
        encoded_cx = image_reg[offsets, 0]
        encoded_cy = image_reg[offsets, 1]
        encoded_width = image_reg[offsets, 2]
        encoded_height = image_reg[offsets, 3]
        encoded_theta_cos = image_reg[offsets, 4]
        encoded_theta_sin = image_reg[offsets, 5]

        l_idx, x, y = get_coord(offsets, map_size, offsets_defaults)
        rs = anchor_sizes[l_idx]
        eps = 1e-6
        image_segments[i, 0] = encoded_cx * rs + (2**(2 + l_idx)) * (x + 0.5)
        image_segments[i, 1] = encoded_cy * rs + (2**(2 + l_idx)) * (y + 0.5)
        image_segments[i, 2] = np.exp(encoded_width) * rs - eps
        image_segments[i, 3] = np.exp(encoded_height) * rs - eps
        image_segments[i, 4] = encoded_theta_cos
        image_segments[i, 5] = encoded_theta_sin

    return image_segments, image_group_indices, image_segments_counts, image_group_indices_all

def decode_image_by_join(node_scores, link_scores, node_threshold,
                         link_threshold, map_size, offsets_defaults):
    node_mask = node_scores[:, POS_LABEL] >= node_threshold
    link_mask = link_scores[:, POS_LABEL] >= link_threshold
    group_mask = np.zeros_like(node_mask, np.int32) - 1
    offsets_pos = np.where(node_mask == 1)[0]

    def find_parent(point):
        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True

        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)

        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)

        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}

        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]

        mask = np.zeros_like(node_mask, dtype=np.int32)
        for i, point in enumerate(offsets_pos):
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    # join by link
    pos_link = 0
    for i, offsets in enumerate(offsets_pos):
        l_idx, x, y = get_coord(offsets, map_size, offsets_defaults)
        neighbours = get_neighbours(l_idx, x, y, map_size, offsets_defaults)
        for n_idx, noffsets in enumerate(neighbours):
            link_value = link_mask[noffsets[1]]
            node_cls = node_mask[noffsets[0]]
            if link_value and node_cls:
                pos_link += 1
                join(offsets, noffsets[0])
    mask = get_all()
    return mask


# decode_segments_links rewrite in python version
def get_coord(offsets, map_size, offsets_defaults):
    if offsets < offsets_defaults[1][0]:
        l_idx = 0
        x = offsets % map_size[0][1]
        y = offsets // map_size[0][1]
    elif offsets < offsets_defaults[2][0]:
        l_idx = 1
        x = (offsets - offsets_defaults[1][0]) % map_size[1][1]
        y = (offsets - offsets_defaults[1][0]) // map_size[1][1]
    elif offsets < offsets_defaults[3][0]:
        l_idx = 2
        x = (offsets - offsets_defaults[2][0]) % map_size[2][1]
        y = (offsets - offsets_defaults[2][0]) // map_size[2][1]
    elif offsets < offsets_defaults[4][0]:
        l_idx = 3
        x = (offsets - offsets_defaults[3][0]) % map_size[3][1]
        y = (offsets - offsets_defaults[3][0]) // map_size[3][1]
    elif offsets < offsets_defaults[5][0]:
        l_idx = 4
        x = (offsets - offsets_defaults[4][0]) % map_size[4][1]
        y = (offsets - offsets_defaults[4][0]) // map_size[4][1]
    else:
        l_idx = 5
        x = (offsets - offsets_defaults[5][0]) % map_size[5][1]
        y = (offsets - offsets_defaults[5][0]) // map_size[5][1]

    return l_idx, x, y

def get_neighbours(l_idx, x, y, map_size, offsets_defaults):
    if l_idx == 0:
        coord = [(0, x - 1, y - 1), (0, x, y - 1), (0, x + 1, y - 1),
                 (0, x - 1, y), (0, x + 1, y), (0, x - 1, y + 1),
                 (0, x, y + 1), (0, x + 1, y + 1)]
    else:
        coord = [(l_idx, x - 1, y - 1),
                 (l_idx, x, y - 1), (l_idx, x + 1, y - 1), (l_idx, x - 1, y),
                 (l_idx, x + 1, y), (l_idx, x - 1, y + 1), (l_idx, x, y + 1),
                 (l_idx, x + 1, y + 1), (l_idx - 1, 2 * x, 2 * y),
                 (l_idx - 1, 2 * x + 1, 2 * y), (l_idx - 1, 2 * x, 2 * y + 1),
                 (l_idx - 1, 2 * x + 1, 2 * y + 1)]
    neighbours_offsets = []
    link_idx = 0
    for nl_idx, nx, ny in coord:
        if is_valid_coord(nl_idx, nx, ny, map_size):
            neighbours_offset_node = offsets_defaults[nl_idx][
                0] + map_size[nl_idx][1] * ny + nx
            if l_idx == 0:
                neighbours_offset_link = offsets_defaults[l_idx][1] + (
                    map_size[l_idx][1] * y + x) * N_LOCAL_LINKS + link_idx
            else:
                off_tmp = (map_size[l_idx][1] * y + x) * (
                    N_LOCAL_LINKS + N_CROSS_LINKS)
                neighbours_offset_link = offsets_defaults[l_idx][
                    1] + off_tmp + link_idx
            neighbours_offsets.append(
                [neighbours_offset_node, neighbours_offset_link, link_idx])
        link_idx += 1
    # [node_offsets, link_offsets, link_idx(0-7/11)]
    return neighbours_offsets


def is_valid_coord(l_idx, x, y, map_size):
    w = map_size[l_idx][1]
    h = map_size[l_idx][0]
    return x >= 0 and x < w and y >= 0 and y < h


""" 
combine_segments_python
"""
def combine_segments_python(segments, group_indices, segment_counts):
    combined_rboxes, combined_counts = combine_segments_batch(segments, group_indices, segment_counts)
    return combined_rboxes, combined_counts


def combine_segments_batch(segments_batch, group_indices_batch,
                           segment_counts_batch):
    batch_size = 1
    combined_rboxes_batch = []
    combined_counts_batch = []
    for image_id in range(batch_size):
        group_count = segment_counts_batch[image_id]
        segments = segments_batch[image_id, :, :]
        group_indices = group_indices_batch[image_id, :]
        combined_rboxes = []
        for i in range(group_count):
            segments_group = segments[np.where(group_indices == i)[0], :]
            if segments_group.shape[0] > 0:
                combined_rbox = combine_segs(segments_group)
                combined_rboxes.append(combined_rbox)
        combined_rboxes_batch.append(combined_rboxes)
        combined_counts_batch.append(len(combined_rboxes))

    max_count = np.max(combined_counts_batch)
    for image_id in range(batch_size):
        if not combined_counts_batch[image_id] == max_count:
            combined_rboxes_pad = (max_count - combined_counts_batch[image_id]
                                   ) * [RBOX_DIM * [0.0]]
            combined_rboxes_batch[image_id] = np.vstack(
                (combined_rboxes_batch[image_id],
                 np.array(combined_rboxes_pad)))

    return np.asarray(combined_rboxes_batch,
                      np.float32), np.asarray(combined_counts_batch, np.int32)


def combine_segs(segs):
    segs = np.asarray(segs)
    assert segs.ndim == 2, 'invalid segs ndim'
    assert segs.shape[-1] == 6, 'invalid segs shape'

    if len(segs) == 1:
        cx = segs[0, 0]
        cy = segs[0, 1]
        w = segs[0, 2]
        h = segs[0, 3]
        theta_sin = segs[0, 4]
        theta_cos = segs[0, 5]
        theta = np.arctan2(theta_sin, theta_cos)
        return np.array([cx, cy, w, h, theta])

    # find the best straight line fitting all center points: y = kx + b
    cxs = segs[:, 0]
    cys = segs[:, 1]

    theta_coss = segs[:, 4]
    theta_sins = segs[:, 5]

    bar_theta = np.arctan2(theta_sins.sum(), theta_coss.sum())
    k = np.tan(bar_theta)
    b = np.mean(cys - k * cxs)

    proj_xs = (k * cys + cxs - k * b) / (k**2 + 1)
    proj_ys = (k * k * cys + k * cxs + b) / (k**2 + 1)
    proj_points = np.stack((proj_xs, proj_ys), -1)

    # find the max distance
    max_dist = -1
    idx1 = -1
    idx2 = -1

    for i in range(len(proj_points)):
        point1 = proj_points[i, :]
        for j in range(i + 1, len(proj_points)):
            point2 = proj_points[j, :]
            dist = np.sqrt(np.sum((point1 - point2)**2))
            if dist > max_dist:
                idx1 = i
                idx2 = j
                max_dist = dist
    assert idx1 >= 0 and idx2 >= 0
    # the bbox: bcx, bcy, bw, bh, average_theta
    seg1 = segs[idx1, :]
    seg2 = segs[idx2, :]
    bcx, bcy = (seg1[:2] + seg2[:2]) / 2.0
    bh = np.mean(segs[:, 3])
    bw = max_dist + (seg1[2] + seg2[2]) / 2.0
    return bcx, bcy, bw, bh, bar_theta


"""工具函数"""
def cal_width(box):
    pd1 = point_dist(box[0], box[1], box[2], box[3])
    pd2 = point_dist(box[4], box[5], box[6], box[7])
    return (pd1 + pd2) / 2

def point_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

def nms_python(boxes):
    boxes = sorted(boxes, key=lambda x: -x[8])
    nms_flag = [True] * len(boxes)
    for i, a in enumerate(boxes):
        if not nms_flag[i]:
            continue
        else:
            for j, b in enumerate(boxes):
                if not j > i:
                    continue
                if not nms_flag[j]:
                    continue
                score_a = a[8]
                score_b = b[8]
                rbox_a = polygon2rbox(a[:8])
                rbox_b = polygon2rbox(b[:8])
                if point_in_rbox(rbox_a[:2], rbox_b) or point_in_rbox(
                        rbox_b[:2], rbox_a):
                    if score_a > score_b:
                        nms_flag[j] = False
    boxes_nms = []
    for i, box in enumerate(boxes):
        if nms_flag[i]:
            boxes_nms.append(box)
    return boxes_nms

def polygon2rbox(polygon):
    x1, x2, x3, x4 = polygon[0], polygon[2], polygon[4], polygon[6]
    y1, y2, y3, y4 = polygon[1], polygon[3], polygon[5], polygon[7]
    c_x = (x1 + x2 + x3 + x4) / 4
    c_y = (y1 + y2 + y3 + y4) / 4
    w1 = point_dist(x1, y1, x2, y2)
    w2 = point_dist(x3, y3, x4, y4)
    h1 = point_line_dist(c_x, c_y, x1, y1, x2, y2)
    h2 = point_line_dist(c_x, c_y, x3, y3, x4, y4)
    h = h1 + h2
    w = (w1 + w2) / 2
    theta1 = np.arctan2(y2 - y1, x2 - x1)
    theta2 = np.arctan2(y3 - y4, x3 - x4)
    theta = (theta1 + theta2) / 2.0
    return [c_x, c_y, w, h, theta]

def point_in_rbox(c, rbox):
    cx0, cy0 = c[0], c[1]
    cx1, cy1 = rbox[0], rbox[1]
    w, h = rbox[2], rbox[3]
    theta = rbox[4]
    dist_x = np.abs((cx1 - cx0) * np.cos(theta) + (cy1 - cy0) * np.sin(theta))
    dist_y = np.abs(-(cx1 - cx0) * np.sin(theta) + (cy1 - cy0) * np.cos(theta))
    return ((dist_x < w / 2.0) and (dist_y < h / 2.0))


def point_line_dist(px, py, x1, y1, x2, y2):
    eps = 1e-6
    dx = x2 - x1
    dy = y2 - y1
    div = np.sqrt(dx * dx + dy * dy) + eps
    dist = np.abs(px * dy - py * dx + x2 * y1 - y2 * x1) / div
    return dist


def rboxes_to_polygons(rboxes):
    """
    Convert rboxes to polygons
    ARGS
        `rboxes`: [n, 5]
    RETURN
        `polygons`: [n, 8]
    """

    theta = rboxes[:, 4:5]
    cxcy = rboxes[:, :2]
    half_w = rboxes[:, 2:3] / 2.
    half_h = rboxes[:, 3:4] / 2.
    v1 = np.hstack([np.cos(theta) * half_w, np.sin(theta) * half_w])
    v2 = np.hstack([-np.sin(theta) * half_h, np.cos(theta) * half_h])
    p1 = cxcy - v1 - v2
    p2 = cxcy + v1 - v2
    p3 = cxcy + v1 + v2
    p4 = cxcy - v1 + v2
    polygons = np.hstack([p1, p2, p3, p4])
    return polygons
