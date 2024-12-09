from datasets import FasterCnnDataset, collate_fn
from model import create_model

import time
import argparse
import numpy as np
import torch
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd


def train(train_dl, val_dl, result_path, epochs, nc, weights):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using pretrained weights: {weights}")

    model = create_model(
        num_classes=nc + 1,  # +1 for background
        trainable_backbone_layers=5,
        weights=weights
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, total trainable params: {total_trainable_params}")
    print(f"Total batches of images: {len(train_dl)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)

    t0 = time.time()

    train_losses = pd.DataFrame(columns=["train/total_loss", "train/cls_loss", "train/box_loss", "train/obj_loss",
                                         "val/total_loss", "val/cls_loss", "val/box_loss", "val/obj_loss"], dtype=float)

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch} out of {epochs} epochs.")

        # Train losses
        e_train_cls_loss        = 0
        e_train_box_loss        = 0
        e_train_obj_loss        = 0
        e_train_rpn_box_loss    = 0
        e_train_loss            = 0
        # Validation losses
        e_val_cls_loss        = 0
        e_val_box_loss        = 0
        e_val_obj_loss        = 0
        e_val_rpn_box_loss    = 0
        e_val_loss            = 0

        ### TRAINING ###
        model.train()

        for i, (images, targets) in enumerate(train_dl):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            e_train_cls_loss += loss_dict["loss_classifier"].cpu().detach().numpy()
            e_train_box_loss += loss_dict["loss_box_reg"].cpu().detach().numpy()
            e_train_obj_loss += loss_dict["loss_objectness"].cpu().detach().numpy()
            e_train_rpn_box_loss += loss_dict["loss_rpn_box_reg"].cpu().detach().numpy()

            loss = sum(v for v in loss_dict.values())
            e_train_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i+1) % 100 == 0:
            #     print(f"Batch {i+1} out of {len(train_dl)} finished with loss: {e_train_loss/(i+1):0.3f}")

        e_train_cls_loss /= len(train_dl)
        e_train_box_loss /= len(train_dl)
        e_train_obj_loss /= len(train_dl)
        e_train_rpn_box_loss /= len(train_dl)
        e_train_loss /= len(train_dl)

        ### VALIDATING ###
        # model.train()  # To get loss easier

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.Dropout2d):
                module.eval()

        # for name, module in model.named_modules():
        #     if hasattr(module, 'training'):
        #         print('{} is training {}'.format(name, module.training))

        for i, (images, targets) in enumerate(val_dl):
            with torch.no_grad():
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                e_val_cls_loss += loss_dict["loss_classifier"].cpu().detach().numpy()
                e_val_box_loss += loss_dict["loss_box_reg"].cpu().detach().numpy()
                e_val_obj_loss += loss_dict["loss_objectness"].cpu().detach().numpy()
                e_val_rpn_box_loss += loss_dict["loss_rpn_box_reg"].cpu().detach().numpy()

                loss = sum(v for v in loss_dict.values())
                e_val_loss += loss.cpu().detach().numpy()

        e_val_cls_loss /= len(val_dl)
        e_val_box_loss /= len(val_dl)
        e_val_obj_loss /= len(val_dl)
        e_val_rpn_box_loss /= len(val_dl)
        e_val_loss /= len(val_dl)


        print(f"Epoch {epoch} finished with losses: --->")
        print(f"\t TRAIN: epoch loss: {e_train_loss:0.3f}, cl: {e_train_cls_loss:0.3f}, box_regr: {e_train_box_loss:0.3f}, obj: {e_train_obj_loss:0.3f}, rpn: {e_train_rpn_box_loss:0.3f}")
        print(f"\t VAL: epoch loss: {e_val_loss:0.3f}, cl: {e_val_cls_loss:0.3f}, box_regr: {e_val_box_loss:0.3f}, obj: {e_val_obj_loss:0.3f}, rpn: {e_val_rpn_box_loss:0.3f}")
        print(f"Time elapsed: {time.time() - t0}")

        # Check bets epoch
        is_best_epoch = np.all(e_val_loss < train_losses["val/total_loss"])
        # Add loses to pandas dataframe
        train_losses.loc[len(train_losses.index)] = [e_train_loss, e_train_cls_loss, e_train_box_loss, e_train_obj_loss,
                                                     e_val_loss, e_val_cls_loss, e_val_box_loss, e_val_obj_loss]

        # Make a plot
        fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
        fax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        names = [x for x in train_losses.columns]
        x = train_losses.index.values.astype(int)
        for i in range(len(names)):
            y = train_losses[names[i]].to_numpy().astype(float)
            fax[i].plot(x, y, marker=".", label="label", linewidth=2, markersize=8)
            fax[i].set_title(names[i], fontsize=12)

        fig.savefig(result_path + "/results.png", dpi=300)
        plt.close()

        # Save *csv file
        train_losses.to_csv(result_path + "/results.csv", sep=",", index=False, header=True)

        # Save weights every
        if epoch % 5 == 0:
            torch.save(model, result_path + f"/epoch{epoch}.pth")

        if is_best_epoch:
            print(f"Epoch {epoch}: save best weights")
            torch.save(model, result_path + f"/best.pth")

    torch.save(model, result_path + f"/last.pth")

    torch.cuda.empty_cache()



def main(args):

    root_path = os.path.dirname(os.path.realpath(__file__))
    result_path = ""
    dataset_path = os.path.expanduser(args.dataset)

    if args.name:
        result_path = os.path.join(root_path, "runs", "train", args.name)
    else:
        for n in range(1, 9999):
            result_path = os.path.join(root_path, "runs", "train", f"exp{n}")
            if not os.path.exists(result_path):  #
                break
            if n == 9998:
                # TODO: Raise exception
                result_path = os.path.join(root_path, "runs", "train", "exp0")
    # Create dir
    os.makedirs(result_path, exist_ok=False)

    label_ids = [int(item) for item in args.label_ids.split(',')] if args.label_ids else []

    weights = args.weights if args.weights else None

    train_dataset = FasterCnnDataset(
        dataset_path=dataset_path,
        dataset_subfolder="train",
        classes=label_ids,
        max_dataset_length=args.dataset_size if args.dataset_size else None,
        use_empty_images=args.use_empty_images
    )
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = FasterCnnDataset(
        dataset_path=dataset_path,
        dataset_subfolder="val",
        classes=label_ids,
        max_dataset_length=args.dataset_size if args.dataset_size else None,
        use_empty_images=args.use_empty_images
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    labels = {}
    with open(os.path.join(dataset_path, os.path.split(dataset_path)[-1]+".yaml"), "r") as f:
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
          f"Used dataset: {dataset_path}\n"
          f"Result location: {result_path}\n"
          f"Batch size: {args.batch_size}, number of epochs: {args.epochs}"
    )

    train(
        train_dl=train_dl,
        val_dl=val_dl,
        result_path=result_path,
        epochs=args.epochs,
        nc=nc,
        weights=weights,
    )



def parse_opt():
    parser = argparse.ArgumentParser(
        description="Usage: train fasterCNN network",
        epilog="python train.py ..."
    )

    default_dataset = "~/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/priede_16_10_24"

    parser.add_argument('--dataset',
                        type=str, help='Path for dataset',
                        default=default_dataset)
    parser.add_argument("--dataset-size", type=int, default=0, help="Max length of dataset") # 0 - no limit

    parser.add_argument('--label_ids', type=str, default="",
                        help="List of labels ids to be used from specific dataset")

    parser.add_argument('--use_empty_images', type=bool, default=True,
                        help="Whether to use empty images")

    parser.add_argument("--name", default=None, help="save to ./runs/train/{exp}")

    parser.add_argument("--epochs", type=int, default=1, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="total batch size for all GPUs")
    parser.add_argument('--weights', type=str, default="",
                        help="Path to custom weights")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_opt()
    main(args)

    #TODO: create and read yaml file from dataset

    # ./run1/train/exp2/last.pth
