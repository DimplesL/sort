# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: tracking.py
@time: 2021/8/17 4:14 下午
@desc:
"""
import os

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

from yolov5.utils.downloads import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
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
            self.model(torch.zeros(1, 3, self.imgsz[0], self.imgsz[1]).to(self.device).type_as(
                next(self.model.parameters())))  # run once

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
        t1 = time_sync()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        det = pred[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        t2 = time_sync()
        return det.numpy(), (t1 - t0, t2 - t1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str,
                        default='/Users/qiuyurui/Desktop/models_file/traffic_4class_yolo5s_720_1280.onnx',
                        help='model.pt path')  # '/Users/qiuyurui/Desktop/models_file/traffic_4class_yolo5s.pt'
    parser.add_argument('--deep_sort_weights', type=str,
                        default='/Users/qiuyurui/Desktop/models_file/res18focalfixepoch=49-step=1849.pt',
                        help='ckpt.t7 path')  # '/Users/qiuyurui/Desktop/models_file/deep_sort_car-ckpt.t7'/Users/qiuyurui/Projects/PycharmProjects/PyTorch_CIFAR10/state_dicts/resnet18.pt
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
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


def cut_image_center_top(img, target_size, ratio_w_half=0.15, ratio_h_down=0.25, ratio_h_top=0.05):
    h, w = img.shape[:2]
    tl_x_shift, tl_y_shift = 0, 0
    if h <= target_size[0] or w <= target_size[1]:
        return img, tl_x_shift, tl_y_shift
    tl_x_shift = int(ratio_w_half * w)
    tl_y_shift = int(ratio_h_top * h)
    return img[tl_y_shift:h - int(ratio_h_down * h), tl_x_shift:w - tl_x_shift, :], tl_x_shift, tl_y_shift


def main(video_path, video_file, save_path):
    # 创建检测器
    args = get_args()
    top_center_cut = True
    img_size = (736, 1280)
    # yolo = Yolo(args, 960)
    yolo = YoloV5ONNX(args.yolo_weights, conf_thres=0.3, iou_thres=0.25)  # traffic-best-6anchor-yolov5s.onnx
    # 创建跟踪器
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.1, dist_thresh=200, size=img_size)
    # tracker = DeepSort(args.deep_sort_weights,
    #                    max_dist=0.2, min_confidence=0.3,
    #                    nms_max_overlap=0.5, max_iou_distance=0.9,
    #                    max_age=30, n_init=2, nn_budget=30,
    #                    use_cuda=True, use_feature=False)

    # 生成多种不同的颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    # 存储中心点
    pts = [deque(maxlen=30) for _ in range(9999)]

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # 检测AB和CD两条直线是否相交
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    # 帧率
    fps_total = 0
    fps_sort = 0

    # ---------------------------------------------------#
    #  读取视频并获取基本信息
    # ---------------------------------------------------#

    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] total {} Frame in video".format(total))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("[INFO] video size :{}".format(size))
        fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
        print("[INFO] video fps :{}".format(fps_cur))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(f"{save_path}/sort_{video_file}", fourcc, fps_cur, (img_size[1], img_size[0]), True)
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

        ori_frame = frame.copy()
        # print(ori_frame.shape)
        cut_frame, tl_x_shift, tl_y_shift = cut_image_center_top(ori_frame, img_size, ratio_w_half=0.15, ratio_h_down=0.25, ratio_h_top=0.05)
        # print(cut_frame.shape)
        frame = cut_frame
        dets, time_used = yolo.predict(frame, img_size=img_size)  # h ,w (576, 1024) (960, 960)

        print(f'predict time used: {time_used}')
        t2 = time.time()

        # todo:track the red and yellow light
        # dets = dets[np.where((dets[:, -1] == 0) | (dets[:, -1] == 2))]

        # if len(dets) != 0:
        #     # tracks = tracker.update(dets)
        #
        # else:
        #     tracks = []
        # if len(dets) == 0:
        #     dets = np.empty((0, 6), dtype=int)
        tracks = tracker.update(dets, frame)

        # tracks = dets

        num = 0
        for track in tracks:
            bbox = track[:4]  # 跟踪框坐标
            indexID = int(track[4])  # 跟踪编号
            # 随机分配颜色
            color = [int(c) for c in COLORS[indexID % len(COLORS)]]
            # 各参数依次是：照片/（左上角，右下角）/颜色/线宽
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
            cv2.putText(frame, f'ID: {str(indexID)} class: {int(track[-1])}', (int(bbox[0]), int(bbox[1] - 10)), 0,
                        5e-1, color, 1)
            # 记录当前帧数量
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

        # 计算帧率
        t3 = time.time()
        print(f"total time: {t3 - t1}")
        fps_total = (fps_total + (1. / (t3 - t1))) / 2
        fps_sort = (fps_sort + (1. / (t3 - t2))) / 2
        # 显示结果
        # cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
        # cv2.putText(frame, str(counter), (20, 90), 0, 0.8, (255, 0, 0), 2)
        # cv2.putText(frame, str(counter_up), (200, 90), 0, 0.8, (0, 255, 0), 2)
        # cv2.putText(frame, str(counter_down), (450, 90), 0, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Current TL Counter: " + str(num), (int(20), int(40)), 0, 5e-1, (0, 255, 0), 2)
        cv2.putText(frame, "FPS total: %f, sort: %f" % (fps_total, fps_sort), (int(20), int(20)), 0, 5e-1, (0, 255, 0), 2)
        cv2.namedWindow("YOLOV5-SORT", 0)
        cv2.resizeWindow('YOLOV5-SORT', 1280, 720)
        writer.write(cv2.resize(frame, (img_size[1], img_size[0])))
        cv2.imshow('YOLOV5-SORT', frame)
        # Q键停止
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def track_videos():
    ori_path = '/Users/qiuyurui/Projects/datas/smart_cycle/lucai_car/360-F18-FOV140-1440P/'
    videos = os.listdir(ori_path)
    save_path = '/Users/qiuyurui/Projects/datas/smart_cycle/lucai_car/360-F18-FOV140-1440P_test3/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for video_file in videos:
        if not video_file.endswith('mp4'):
            continue
        video_path = os.path.join(ori_path, video_file)
        # ("/Users/qiuyurui/Projects/datas/bilibili_dashcam/249852744-1-192.mp4.mp4")
        # #('/Users/qiuyurui/Projects/datas/smart_cycle/lucai_car/400w-banzai/part2-2021.8.24.ts')
        main(video_path, video_file, save_path)


if __name__ == '__main__':
    video_path = '/Users/qiuyurui/Projects/datas/bilibili_dashcam/'
    video_file = '249852744-1-192.mp4.mp4'
    save_path = './'
    # main(os.path.join(video_path, video_file), video_file, save_path)
    track_videos()