from datasets import CustomDataset, collate_fn
from model import create_model

import numpy as np
import torch
import os

class CustomModelTraining:
    def __init__(self, classes, device, weights=None):
        self.classes = classes
        self.num_classes = len(self.classes)
        self.device = device
        self.weights = weights

        self.train_dataset = CustomDataset(
            dataset_path="datasets/2024-09-18 11:32:03.928673",
            annot_format="faster",
            classes=[2],
            use_train=True,
            use_test=False,
            use_val=False,
            max_dataset_length=None
        )

        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        )

        print(f"Length of training dataset: {len(self.train_dataset)}")

        if weights is None:
            self.model = create_model(self.num_classes,
                                  trainable_backbone_layers=2).to(self.device)
        else:
            self.model = torch.load(self.weights).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, nesterov=True)
        self.num_epochs = 10

        # Losses
        self.train_loss_list = []


    def train_one_epoch(self):
        self.model.train() # Set in training mode
        epoch_loss = 0

        loss_list = []

        for i, (images, targets) in enumerate(self.train_dl):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())
            loss_list.append(loss)
            epoch_loss += loss.cpu().detach().numpy()

            if len(loss_list) == 10:
                self.optimizer.zero_grad()
                sum(loss_list).backward()
                self.optimizer.step()
                loss_list.clear()


            if i+1 % 10 == 0:
                print(sum(loss_list))
        print(epoch_loss)

    def train_one_epoch1(self):
        self.model.train()  # Set in training mode
        epoch_loss = 0
        print(f"Len: {len(self.train_dl)}")
        for i, (images, targets) in enumerate(self.train_dl):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())
            epoch_loss += loss.cpu().detach().numpy()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(loss)
        print(epoch_loss)


if __name__ == "__main__":
    #TODO: create and read yaml file from dataset
    CLASSES = ["__background__",
               "egle"]
    device = "cuda" # cpu

    torch.cuda.empty_cache()

    cmt = CustomModelTraining(CLASSES, device)

    for j in range(cmt.num_epochs):
        cmt.train_one_epoch1()
        # Save model
        torch.save(cmt.model, f"./saved_models/egle_{j}.pth")


    #print(cmt.model)
