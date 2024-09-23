from datasets import CustomDataset, collate_fn
from model import create_model

import numpy as np
import torch, torchvision
import os
import cv2 as cv
import matplotlib.pyplot as plt
import argparse

class CustomModelDetect:
    def __init__(self, model_weights):

        self.num_classes = 2
        # self.model = create_model(self.num_classes).to("cpu")
        self.model = torch.load(model_weights).to("cpu")

    def inference(self, img):
        self.model.eval()

        images = [img]
        out = self.model(images)
        return out



if __name__ == "__main__":
    #TODO: create and read yaml file from dataset ??
    #TODO: do label mapping
    #TODO: add argparser

    device = "cuda" # cpu
    torch.cuda.empty_cache()

    cmi = CustomModelDetect("./saved_models/egle_9.pth")

    dataset = CustomDataset(
        dataset_path="datasets/2024-09-18 11:32:03.928673",
        annot_format="faster",
        classes=[2],
        use_train = True,
        use_test = False,
        use_val = False,
        max_dataset_length=None
    )

    for ii in range(3):

        img, labels = dataset[ii]

        img_orig = (img.permute(1, 2, 0).cpu().detach().numpy() * 255)
        img_orig = np.ascontiguousarray(img_orig, dtype=np.uint8)

        output = cmi.inference(img.to("cpu"))
        out_boxes = output[0]["boxes"].cpu().detach()
        out_scores = output[0]["scores"].cpu().detach()
        keep = torchvision.ops.nms(out_boxes, out_scores, 0.45)
        # print(keep, out_boxes.shape)
        out_boxes = torch.index_select(out_boxes, 0, keep)
        out_scores = torch.index_select(out_scores, 0, keep)
        # print(keep, out_boxes.shape)

        print(out_scores)
        for box, score in zip(out_boxes, out_scores):
            if score < 0.45:
                continue
            box = box.numpy().astype("int")
            img_orig = cv.rectangle(
                img_orig,
                (box[0], box[1], box[2]-box[0], box[3]-box[1]),
                (255, 0, 0), #rgb
                7
            )

        orig_boxes = labels["boxes"]
        for box in orig_boxes:
            box = box.numpy().astype("int")
            img_orig = cv.rectangle(img_orig,
                            (box[0], box[1], box[2]-box[0], box[3]-box[1]),
                            (0, 255, 0),
                            7)

        plt.imshow(img_orig)
        plt.show()


