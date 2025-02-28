import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import glob
from models.faceland import FaceLanndInference
from hdface.hdface import hdface_detector


class FaceLandmarkInference:
    def __init__(self, weight_file, use_cuda=False):
        self.det = hdface_detector(use_cuda=use_cuda)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        checkpoint = torch.load(weight_file, map_location=self.device)
        self.plfd_backbone = FaceLanndInference().to(self.device)
        self.plfd_backbone.load_state_dict(checkpoint)
        self.plfd_backbone.eval()
        self.plfd_backbone.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.keypoints = [0, 7, 25, 32, 57, 96, 97]

    def detect(self, img):
        image = cv2.imread(os.path.join(image_dir, img))
        height, width = image.shape[:2]
        print(height, width)
        with torch.no_grad():
            img_det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.det.detect_face(img_det)
        return result

    def predict(self, image):
        # image = img.copy()
        height, width = image.shape[:2]
        # print(height, width)
        img_det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.det.detect_face(img_det)

        if len(result) != 1:
            return result, None, None

        box = result[0]['box']
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size_w = int(max([w, h]) * 0.8)
        size_h = int(max([w, h]) * 0.8)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size_w // 2
        x2 = x1 + size_w
        y1 = cy - int(size_h * 0.4)
        y2 = y1 + size_h

        left = 0
        top = 0
        bottom = 0
        right = 0
        if x1 < 0:
            left = -x1
        if y1 < 0:
            top = -y1
        if x2 >= width:
            right = x2 - width
        if y2 >= height:
            bottom = y2 - height

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))

        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        cropped = image[y1:y2, x1:x2]
        # print(top, bottom, left, right)
        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transform(input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            landmarks = self.plfd_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]
        pre_landmark = pre_landmark.astype(np.int32)

        eye_l = (x1 - left + pre_landmark[96][0], y1 - bottom + pre_landmark[96][1])
        eye_r = (x1 - left + pre_landmark[97][0], y1 - bottom + pre_landmark[97][1])
        k_coef = (eye_r[1] - eye_l[1]) / (eye_r[0] - eye_l[0])
        nose = (x1 - left + pre_landmark[57][0], y1 - bottom + pre_landmark[57][1])
        chin = (x1 - left + pre_landmark[17][0], y1 - bottom + pre_landmark[17][1])

        pre_landmark_orig = pre_landmark.copy()
        pre_landmark_orig[:, 0] = x1 - left + pre_landmark[:, 0]
        pre_landmark_orig[:, 1] = y1 - bottom + pre_landmark[:, 1]

        # for i in range(len(pre_landmark_orig)):
        #     cv2.circle(image, (pre_landmark_orig[i][0], pre_landmark_orig[i][1]), 2, (255, 0, 255), 2)
        #     cv2.imshow('pre', image)
        #     cv2.waitKey(0)

        return result, pre_landmark_orig, cropped

        # left_point = (0, int(nose[1] - k_coef * nose[0]))
        # right_point = (width, int(k_coef * width - k_coef * nose[0] + nose[1]))
        # bottom_point = (int((height + k_coef * nose[0] - nose[1])/(k_coef + 1e-6)), height)
        # contours = [left_point, right_point, (width, height), (0, height)]
        # if left_point[1] > height:
        #     left_point = bottom_point
        #     contours = [left_point, right_point, (width, height)]
        # if right_point[1] > height:
        #     right_point = bottom_point
        #     contours = [left_point, right_point, (0, height)]
        #
        # contours = np.array(contours)
        # cv2.fillPoly(image, [contours], (0, 0, 0))
        #
        # for (x, y) in [eye_l, eye_r, nose, left_point, right_point]:
        #     print(x, y)
        #     cv2.circle(image, (x, y), 2, (255, 0, 255), 2)
        #     cv2.imshow('pre', image)
        #     cv2.waitKey(0)
        #
        # # save_file = os.path.join(save_dir, img)
        # cv2.imwrite("result.jpg", image)
        #
        # return cropped




if __name__ == "__main__":

    image_dir = "/Users/a58/Downloads/c4200_2/38"
    save_dir = "/Users/a58/Downloads/facelandmarks/results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    FaceLandmark = FaceLandmarkInference()
    for file in sorted(os.listdir(image_dir)):
        if file.startswith('.'): continue
        file_path = os.path.join(image_dir, file)
        FaceLandmark.predict(file_path)
