import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import Counter
import cv2
from glob import glob
from tqdm import tqdm
from termcolor import colored

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
from CustomVOCDataset import CustomVOCDataset



from utils.map import mean_average_precision
from YoloLoss import YoloLoss
from Yolov1 import Yolov1
from utils.npms import non_max_suppression

class_mapping = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


# Set the random seed for reproducibility.
seed = 123
torch.manual_seed(seed)

# Hyperparameters and configurations
# Learning rate for the optimizer.
LEARNING_RATE = 2e-5
# Specify whether to use "cuda" (GPU) or "cpu" for training.
DEVICE = "cuda"
# Originally 64 in the research paper, but using a smaller batch size due to GPU limitations.
BATCH_SIZE = 16
# Number of training epochs.
EPOCHS = 200
# Number of worker processes for data loading.
NUM_WORKERS = 2
# If True, DataLoader will pin memory to transfer data to the GPU faster.
PIN_MEMORY = True
# If False, the training process will not load a pre-trained model.
LOAD_MODEL = False
# Specify the file name for the pre-trained model if LOAD_MODEL is True.
LOAD_MODEL_FILE = "final_yolov1.pth.tar"


WIDTH = 448
HEIGHT = 448

def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      A.ToGray(p=0.01),
                      A.HorizontalFlip(p=0.2),
                      A.VerticalFlip(p=0.2),
                      A.Resize(height=WIDTH, width=HEIGHT, p=1),
                    #   A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=WIDTH, width=HEIGHT, p=1.0),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                      )


# Function to save model and optimizer state to a checkpoint file
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Function to load model and optimizer state from a saved checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Function to get predicted and true bounding boxes from the model's output and ground truth data
def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # Ensure the model is in evaluation mode before obtaining bounding boxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # Convert multiple boxes to 0 if predicted
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


# Function to convert cell-based boxes to format [x_center, y_center, width, height]
def convert_cellboxes(predictions, S=7):
    """
    Convert output boxes from YOLO with grid size S to image scale,
    instead of grid scale. This implementation uses loops for readability,
    as vectorized approaches can be less readable.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

# Function to convert cell-based boxes to a list of boxes for each example in the batch
def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def get_bboxes_training(
    outputs,
    labels,
    iou_threshold=0.5,
    threshold=0.4,
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # Ensure the model is in evaluation mode before obtaining bounding boxes
    train_idx = 0

    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(outputs)

    for idx in range(outputs.shape[0]):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[idx]:
            # Convert multiple boxes to 0 if predicted
            if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)

        train_idx += 1

    return all_pred_boxes, all_true_boxes 

def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    mean_loss = []
    mean_mAP = []

    total_batches = len(train_loader)
    display_interval = total_batches // 5  # Update after 20% of the total batches.

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

        if batch_idx % display_interval == 0 or batch_idx == total_batches - 1:
            print(f"Epoch: {epoch:3} \t Iter: {batch_idx:3}/{total_batches:3} \t Loss: {loss.item():3.10f} \t mAP: {mAP.item():3.10f}")

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"Train \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}", 'green'))

    return avg_mAP

def test_fn(test_loader, model, loss_fn, epoch):
    model.eval()
    mean_loss = []
    mean_mAP = []

    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"Test \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}", 'yellow'))

    model.train()

    return avg_mAP

# Main function.
def train():
    # Initialize the YOLOv1 model, Adam optimizer, and YOLO loss function.
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YoloLoss()

    # If requested, load a previously saved model.
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Create training and validation datasets with specified parameters.
    train_dataset = CustomVOCDataset(root='./data',
                                     class_mapping=class_mapping,
                                     year='2012',
                                     image_set='trainval',
                                     download=False,
                                     transform=get_train_transforms()
                                     )

    test_dataset = CustomVOCDataset(root='./data',
                                    class_mapping=class_mapping,
                                    year='2012',
                                    image_set='val',
                                    download=False,
                                    transform=get_valid_transforms()
                                    )

    # Create DataLoader for training and validation datasets with specified settings.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    best_mAP_train = 0
    best_mAP_test = 0

    # Iterate through each epoch and compute predicted and target bounding boxes using the get_bboxes function.
    for epoch in range(EPOCHS):
        train_mAP = train_fn(train_loader, model, optimizer, loss_fn, epoch)
        test_mAP = test_fn(test_loader, model, loss_fn, epoch)

        # Update the best mAP for train and test
        if train_mAP > best_mAP_train:
            best_mAP_train = train_mAP

        if test_mAP > best_mAP_test:
            best_mAP_test = test_mAP

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


    print(colored(f"Best Train mAP: {best_mAP_train:3.10f}", 'green'))
    print(colored(f"Best Test mAP: {best_mAP_test:3.10f}", 'yellow'))



train()