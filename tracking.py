# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: tracking.py
@time: 2021/8/17 4:14 下午
@desc:
"""
from sort import Sort
import time
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# sys.path.insert(-2, '../../PycharmProjects/yolov5')
sys.path.append('..')
sys.path.append('../yolov5')
sys.path.append('../../PycharmProjects/yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from yolov5_onnx import YoloV5ONNX
from collections import deque
from deep_sort import DeepSort


# ---------------------------------------------------#
#  初始化
# ---------------------------------------------------#

class Yolo:
    def __init__(self, opt, imgsz):
        self.device = select_device(opt.device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.half = half
        self.opt = opt
        # Load model
        self.model = attempt_load(opt.yolo_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if half:
            self.model.half()  # to FP16
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz[0], self.imgsz[1]).to(self.device).type_as(next(self.model.parameters())))  # run once

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @torch.no_grad()
    def predict(self, img0):
        t0 = time.time()
        img = self.letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        det = pred[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        t2 = time_synchronized()
        return det.numpy(), (t1 - t0, t2 - t1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str,
                        default='/Users/qiuyurui/Desktop/models_file/traffic_4class_yolo5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='/Users/qiuyurui/Desktop/models_file/deep_sort_car-ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    return args


def main():
    # 创建检测器
    args = get_args()
    # yolo = Yolo(args, 960)
    yolo = YoloV5ONNX('/Users/qiuyurui/Desktop/models_file/traffic_4class_yolo5s.onnx')  # traffic-best-6anchor-yolov5s.onnx
    # 创建跟踪器
    # tracker = Sort()
    tracker = DeepSort(args.deep_sort_weights,
                       max_dist=0.2, min_confidence=0.3,
                       nms_max_overlap=0.5, max_iou_distance=0.7,
                       max_age=70, n_init=3, nn_budget=100,
                       use_cuda=True)

    # 生成多种不同的颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    # 存储中心点
    pts = [deque(maxlen=30) for _ in range(9999)]
    # 帧率
    fps_total = 0
    fps_sort = 0

    # ---------------------------------------------------#
    #  虚拟线圈统计车流量
    # ---------------------------------------------------#
    # 虚拟线圈
    line = [(0, 100), (1500, 100)]

    # AC = ((C[0] - A[0]), (C[1] - A[1]))
    # AB = ((B[0] - A[0]), (B[1] - A[1]))
    # 计算由A，B，C三点构成的向量AC，AB之间的关系
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # 检测AB和CD两条直线是否相交
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    # 车辆总数
    counter = 0
    # 正向车道的车辆数据
    counter_up = 0
    # 逆向车道的车辆数据
    counter_down = 0

    # ---------------------------------------------------#
    #  读取视频并获取基本信息
    # ---------------------------------------------------#
    cap = cv2.VideoCapture("/Users/qiuyurui/Projects/datas/bilibili_dashcam/249852744-1-192.mp4.mp4")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] total {} Frame in video".format(total))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("[INFO] video size :{}".format(size))
        fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
        print("[INFO] video fps :{}".format(fps_cur))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter("./output.mp4", fourcc, fps_cur, size, True)
    except:
        print("[INFO] could not determine in video")

    # ---------------------------------------------------#
    #  逐帧检测并追踪
    # ---------------------------------------------------#
    tracks = []
    frame_sample = 5
    i = 0
    while True:
        if not i % frame_sample == 0:
            cap.grab()
            i += 1
            continue
        else:
            i = 0
        (ret, frame) = cap.read()
        i += 1
        if not ret:
            break
        t1 = time.time()

        dets, time_used = yolo.predict(frame)
        print(f'predict time used: {time_used}')
        t2 = time.time()
        if len(dets) == 0:
            pass
        else:
            # tracks = tracker.update(dets)
            tracks = tracker.update(dets, frame)

        num = 0
        for track in tracks:
            bbox = track[:4]  # 跟踪框坐标
            indexID = int(track[4])  # 跟踪编号
            # 随机分配颜色
            color = [int(c) for c in COLORS[indexID % len(COLORS)]]
            # 各参数依次是：照片/（左上角，右下角）/颜色/线宽
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
            cv2.putText(frame, str(indexID) + f'_class: {track[-1]}', (int(bbox[0]), int(bbox[1] - 10)), 0, 5e-1, color, 1)
            # 记录当前帧的车辆数
            num += 1
            # 检测框中心(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            pts[indexID].append(center)
            cv2.circle(frame, (center), 1, color, 2)
            # 显示运动轨迹
            for j in range(1, len(pts[indexID])):
                if pts[indexID][j - 1] is None or pts[indexID][j] is None:
                    continue
                cv2.line(frame, (pts[indexID][j - 1]), (pts[indexID][j]), color, 2)
            # 虚拟线圈计数
            if len(pts[indexID]) >= 2:
                p1 = pts[indexID][-2]
                p0 = pts[indexID][-1]
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                    if p1[1] > p0[1]:
                        counter_down += 1
                    else:
                        counter_up += 1

        # 计算帧率
        t3 = time.time()
        print(f"total time: {t3 - t1}")
        fps_total = (fps_total + (1. / (t3 - t1))) / 2
        fps_sort = (fps_sort + (1. / (t3 - t2))) / 2
        # 显示结果
        cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
        cv2.putText(frame, str(counter), (20, 90), 0, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, str(counter_up), (200, 90), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, str(counter_down), (450, 90), 0, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Current TL Counter: " + str(num), (int(20), int(40)), 0, 5e-1, (0, 255, 0), 2)
        cv2.putText(frame, "FPS total: %f, sort: %f" % (fps_total, fps_sort), (int(20), int(20)), 0, 5e-1, (0, 255, 0),
                    2)
        cv2.namedWindow("YOLOV5-SORT", 0)
        cv2.resizeWindow('YOLOV5-SORT', 1280, 720)
        writer.write(frame)
        cv2.imshow('YOLOV5-SORT', frame)
        # Q键停止
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
