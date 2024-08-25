import torch
from utils.iou import intersection_over_union


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
  # bboxes (list): [[class_pred, prob_score, x1, y1, x2, y2]].
  bboxes = [box for box in bboxes if box[1] > threshold]
  bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
  
  bboxes_keep = []
  while bboxes:
    chosen_box = bboxes.pop(0)
    bboxes = [
        box for box in bboxes
        if box[0] != chosen_box[0]
        or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), 
                                                box_format=box_format,) < iou_threshold
    ]
    bboxes_keep.append(chosen_box)

  return bboxes_keep