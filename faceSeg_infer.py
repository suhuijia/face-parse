import os
import argparse
import json
import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet
from utils.common import ATTRIBUTES, COLOR_LIST, letterbox, vis_parsing_maps


def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


@torch.no_grad()
def inference(config):
    output_path = config.output
    input_path = config.input
    weight = config.weight
    model = config.model

    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name=model)
    model.to(device)

    if os.path.exists(weight):
        model.load_state_dict(torch.load(weight, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight})")

    if os.path.isfile(input_path):
        input_path = [input_path]

    model.eval()

    for filename in os.listdir(input_path):
        if not filename.endswith('png'): continue
        file_path = os.path.join(input_path, filename)
        image = Image.open(file_path).convert("RGB")
        w, h = image.size
        print(f"Processing image: {file_path}")

        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)

        output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        blended_image, segmentation_mask_color = vis_parsing_maps(resized_image, predicted_mask, save_image=True, save_path=os.path.join(output_path, filename))
        segmentation_mask_color = cv2.resize(segmentation_mask_color, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_path, filename), segmentation_mask_color)


class FaceSeg():
    def __init__(self, weight_file, modelname, use_cuda=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        num_classes = 19
        self.model = BiSeNet(num_classes, backbone_name=modelname)
        self.model.to(self.device)

        if os.path.exists(weight_file):
            self.model.load_state_dict(torch.load(weight_file, map_location=self.device))
        else:
            raise ValueError(f"Weights not found from given path ({weight_file})")

        self.model.eval()

    @torch.no_grad()
    def predict(self, face_img, save_path, save_img=False):
        # filename = os.path.basename(file_path)
        # image = Image.open(file_path).convert("RGB")
        image = Image.fromarray(face_img)
        w, h = image.size
        resized_image = image.resize((256, 256), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(self.device)

        output = self.model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        blended_image, segmentation_mask_color, segmentation_mask_index = vis_parsing_maps(resized_image, predicted_mask)
        if save_img:
            cv2.imwrite(save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        segmentation_mask_color = cv2.resize(segmentation_mask_color, (w, h), interpolation=cv2.INTER_NEAREST)
        segmentation_mask_index = cv2.resize(segmentation_mask_index, (w, h), interpolation=cv2.INTER_NEAREST)

        return segmentation_mask_color, segmentation_mask_index


def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model", type=str, default="resnet34", help="model name, i.e resnet18, resnet34")
    parser.add_argument("--weight", type=str, default="./weights/resnet34.pt", help="path to trained model")
    parser.add_argument("--input", type=str, default="/Users/a58/Downloads/test", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="/Users/a58/Downloads/test_result", help="path to save model outputs")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # inference(config=args)
    FP = FaceSeg(weight_file=args.weight, modelname=args.model, use_cuda=False)

    json_file = ""
    image_root = ""
    with open(json_file, 'r', encoding='utf-8') as f:
        json_file = json.load(f)

    for ele in json_file:
        save_ele = ele.replace("frames_ratio", "frames_ratio_mask")
        image_file = os.path.join(image_root, ele)
        save_file = os.path.join(image_root, save_ele)
        save_dir = os.path.dirname(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mask_image = FP.predict(image_file)
        cv2.imwrite(save_file, mask_image)