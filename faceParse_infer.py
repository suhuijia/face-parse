# -*- encoding: utf-8 -*-
# @Author: SHJ
# @Time: 2025-02-27 17:39
import os
import argparse
import numpy as np
import cv2
from faceSeg_infer import FaceSeg
from faceLandmark_infer import FaceLandmarkInference


# 定义一个函数来计算头部姿态
def calculate_head_pose(shape, image):
    # 获取特定关键点的索引 60, 72, 54, 76, 82
    image_points = np.array([
        (shape[54][0], shape[54][1]),  # 鼻子
        (shape[16][0], shape[16][1]),
        (shape[60][0], shape[60][1]),   # 左眼角
        (shape[72][0], shape[72][1]),  # 右眼角
        (shape[76][0], shape[76][1]),  # 左嘴角
        (shape[82][0], shape[82][1]),  # 右嘴角
    ], dtype='double')

    # 3D模型点（根据标准的头部模型）
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 鼻子
        (0.0, -330.0, -65.0),         # 下巴
        (-225.0, 170.0, -135.0),     # 左眼角
        (225.0, 170.0, -135.0),      # 右眼角
        (-150.0, -150.0, -125.0),    # 左嘴角
        (150.0, -150.0, -125.0)      # 右嘴角
    ], dtype='double')

    # 相机焦距
    focal_length = image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)

    # 相机内参矩阵
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    # 假设没有镜头畸变
    dist_coeffs = np.zeros((4, 1))

    # 使用OpenCV的solvePnP函数来计算旋转和平移向量
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # 使用Rodrigues变换将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))

    # 计算欧拉角（俯仰角，偏航角，滚动角）
    pitch, yaw, roll = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    return pitch, yaw, roll


def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model_seg", type=str, default="resnet34", help="model name, i.e resnet18, resnet34")
    parser.add_argument("--weight_seg", type=str, default="./weights/resnet34.pt", help="")
    parser.add_argument("--weight_landmark", type=str, default="./weights/faceland.pth", help="")
    parser.add_argument("--use_cuda", type=bool, default=False, help="")
    parser.add_argument("--save_dir", type=str, default="/Users/a58/Downloads/test_result", help="")
    return parser.parse_args()


class FaceParser:
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.faceSeg = FaceSeg(weight_file=args.weight_seg, modelname=args.model_seg, use_cuda=args.use_cuda)
        self.faceLandmark = FaceLandmarkInference(weight_file=args.weight_landmark, use_cuda=args.use_cuda)

    def inference(self, image, img_path):
        filename = os.path.basename(img_path)
        only_person = 0
        open_mouth = 0
        rotation_head = 0
        result, pre_landmark_orig, cropped = self.faceLandmark.predict(image)
        ## 头部关键点可视化
        idxs = [16, 60, 72, 54, 76, 82, 96, 97]
        # for idx in idxs:
        #     cv2.circle(image, (pre_landmark_orig[idx][0], pre_landmark_orig[idx][1]), 2, (0, 0, 255), -1)
        #     cv2.imshow("landmark", image)
        #     cv2.waitKey(0)

        if len(result) == 1:
            only_person = 1

        if cropped is not None:
            mask_img, mask_index = self.faceSeg.predict(cropped[:,:,::-1], save_path=os.path.join(self.save_dir, filename))
            # blended_image = cv2.addWeighted(cropped, 0.6, mask_img, 0.4, 0)
            # cv2.imshow("blended_image", blended_image)
            # cv2.waitKey(0)

            ## 判断张嘴闭嘴，通过嘴部张开区域分割结果判别
            mouth_mask = mask_index > 0
            open_mask = mask_index == 11
            # cv2.imshow("mouth_mask", (mouth_mask*255).astype("uint8"))
            # cv2.imshow("open_mask", (open_mask*255).astype("uint8"))
            open_ratio = np.sum(open_mask) / np.sum(mouth_mask)
            print("open_ratio", open_ratio)
            if open_ratio > 0.05:
                open_mouth = 1

            ## 计算头部旋转角度
            pitch, yaw, roll = calculate_head_pose(shape=pre_landmark_orig, image=image)
            print(f"pitch: {pitch}   yaw: {yaw}   roll: {roll}")
            pitch_abs = min(180 - abs(pitch), abs(pitch))
            yaw_abs = min(180 - abs(yaw), abs(yaw))
            roll_abs = min(180 - abs(roll), abs(roll))
            if max(pitch_abs, yaw_abs, roll_abs) > 10:
                rotation_head = 1

        return only_person, open_mouth, rotation_head


def test_image():
    ## 单图测试
    img_file = "/Users/a58/Downloads/80/00011.jpg"
    image = cv2.imread(img_file)
    faceParser.inference(image, img_file)


def test_dir():
    ## 文件夹测试
    image_dir = "/Users/a58/Downloads/80"
    for file in sorted(os.listdir(image_dir)):
        if not file.endswith('.jpg'): continue
        print(file)
        img_path = os.path.join(image_dir, file)
        image = cv2.imread(img_path)
        only_person, open_mouth, rotation_head = faceParser.inference(image, img_path)


def test_video():
    ## 视频测试
    video_file = "/Users/a58/Downloads/0.mp4"
    image_dir = "/Users/a58/Downloads/test_result"
    cap = cv2.VideoCapture(video_file)
    frame_num = 0
    only_person_num, open_mouth_num, rotation_head_num = 0, 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        img_path = os.path.join(image_dir, str(frame_num).zfill(5) + '.jpg')
        only_person, open_mouth, rotation_head = faceParser.inference(frame, img_path)
        only_person_num += only_person
        open_mouth_num += open_mouth
        rotation_head_num += rotation_head
        print(f"only_person: {only_person}   open_mouth: {open_mouth}   rotation_head: {rotation_head}")
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    only_person_ratio = only_person_num / frame_num
    open_mouth_ratio = open_mouth_num / only_person_num
    rotation_head_ratio = rotation_head_num / only_person_num
    print(f"only_person_ratio: {only_person_ratio}", only_person_num, frame_num)
    print(f"open_mouth_ratio: {open_mouth_ratio}", open_mouth_num)
    print(f"rotation_head_ratio: {rotation_head_ratio}", rotation_head_num)


if __name__ == "__main__":

    args = parse_args()
    faceParser = FaceParser(args)

    # ## 单图测试
    # test_image()
    #
    # ## 文件夹测试
    # test_dir()

    ## 视频测试
    test_video()