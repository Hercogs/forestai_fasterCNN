from datasets import FasterCnnDataset, collate_fn
from model import create_model

import time
import argparse
import numpy as np
import torch
import os
import yaml

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

def train1(train_dl, val_dl, result_path, epochs, nc):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    print(f"Using device: {device}")

    model = create_model(
        num_classes=nc + 1,  # +1 for background
        trainable_backbone_layers=5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, total trainable params: {total_trainable_params}")
    print(f"Total batches of images: {len(train_dl)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)

    t0 = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} out of {epochs} epochs.")
        model.train()

        loss_classifier = 0
        loss_box_reg = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0
        epoch_loss = 0

        for i, (images, targets) in enumerate(train_dl):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            loss_classifier += loss_dict["loss_classifier"].cpu().detach().numpy()
            loss_box_reg += loss_dict["loss_box_reg"].cpu().detach().numpy()
            loss_objectness += loss_dict["loss_objectness"].cpu().detach().numpy()
            loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].cpu().detach().numpy()

            loss = sum(v for v in loss_dict.values())
            epoch_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Batch {i+1} out of {len(train_dl)} finished with loss: {epoch_loss/(i+1):0.3f}")

        print(f"Epoch {epoch+1} finished with losses ---> epoch loss: {epoch_loss},\n"
                f"\tepoch loss: {epoch_loss:0.3f}, cl: {loss_classifier:0.3f}, box_regr: {loss_box_reg:0.3f}, obj: {loss_objectness:0.3f}, rpn: {loss_rpn_box_reg:0.3f}")
        print(f"Time elapsed: {time.time() - t0}")

        # Save weights
        torch.save(model, result_path + f"/w{epoch+1}.pth")



def main(args):

    root_path = os.path.dirname(os.path.realpath(__file__))
    result_path = ""

    if args.name:
        result_path = os.path.join(root_path, "runs", "train", args.name)
    else:
        for n in range(1, 9999):
            result_path = os.path.join(root_path, "runs", "train", f"exp{n}")
            if not os.path.exists(result_path):  #
                break
            if n == 9998:
                result_path = os.path.join(root_path, "runs", "train", "exp0")
    # Create dir
    os.makedirs(result_path, exist_ok=False)

    label_ids = [int(item) for item in args.label_ids.split(',')] if args.label_ids else []

    train_dataset = FasterCnnDataset(
        dataset_path=args.dataset,
        dataset_subfolder="train",
        classes=label_ids,
        max_dataset_length=None,
        use_empty_images=args.use_empty_images
    )
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = FasterCnnDataset(
        dataset_path=args.dataset,
        dataset_subfolder="val",
        classes=label_ids,
        max_dataset_length=None,
        use_empty_images=args.use_empty_images
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    labels = {}
    with open(os.path.join(args.dataset, os.path.split(args.dataset)[-1]+".yaml"), "r") as f:
        data_loaded = yaml.safe_load(f)
        names = data_loaded["names"]

        if label_ids:
            for label in label_ids:
                labels[label] = names[label]
        else:
            labels = names

    nc = len(labels)

    print("Starting to train FasterCNN network.\n"
          f"Number of classes: {nc}\n\t{labels}\n"
          f"Used dataset: {args.dataset}\n"
          f"Result location: {result_path}\n"
          f"Batch size: {args.batch_size}, number of epochs: {args.epochs}"
    )

    train1(
        train_dl=train_dl,
        val_dl=val_dl,
        result_path=result_path,
        epochs=args.epochs,
        nc=nc
    )



def parse_opt():
    parser = argparse.ArgumentParser(
        description="Usage: train fasterCNN network",
        epilog="python train.py ..."
    )

    default_dataset = "/home/hercogs/Desktop/Droni/git_repos/forestai_dataset_generation/datasets/priede_test"

    parser.add_argument('--dataset',
                        type=str, help='Path for dataset',
                        default=default_dataset)
    # parser.add_argument('--dataset_subfolder', type=str, default="train")
    parser.add_argument('--label_ids', type=str, default="",
                        help="List of labels ids to be used from specific dataset")

    parser.add_argument('--use_empty_images', type=bool, default=False,
                        help="Whether to use empty images")

    parser.add_argument("--name", default=None, help="save to ./runs/train/{exp}")

    parser.add_argument("--epochs", type=int, default=2, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_opt()
    main(args)

    # #TODO: create and read yaml file from dataset
    # CLASSES = ["__background__",
    #            "egle"]
    # device = "cuda" # cpu
    #
    # torch.cuda.empty_cache()
    #
    # cmt = CustomModelTraining(CLASSES, device)
    #
    # for j in range(cmt.num_epochs):
    #     cmt.train_one_epoch1()
    #     # Save model
    #     torch.save(cmt.model, f"./saved_models/egle_{j}.pth")


    #print(cmt.model)
