import torch
import torchvision
import numpy as np
from utils.convert_to_yolo_format import convert_to_yolo_format

class CustomVOCDataset(torchvision.datasets.VOCDetection):
  def __init__(self, class_mapping, S=7, B=2, C=20, transform=None, **kwargs):
    super(CustomVOCDataset, self).__init__(**kwargs)
    self.S = S #Grid SxS
    self.B = B #Number of BoundingBoxes
    self.C = C #Number of classes
    self.class_mapping = class_mapping
    self.transform = transform
  
  def __getitem__(self, index: int):
    image, target = super(CustomVOCDataset, self).__getitem__(index)
    
    #Custom
    boxes = convert_to_yolo_format(target, self.class_mapping)
    cords = boxes[:, 1:]
    labels = boxes[:, 0]

    if self.transform:
      sample = {
          'image': np.array(image),
          'bboxes': cords,
          'labels': labels
      }
      sample = self.transform(**sample)
      image = sample['image']
      boxes = sample['bboxes']
      labels = sample['labels']

    
    label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    image  = torch.as_tensor(image, dtype=torch.float32)

    for box, class_label in zip(boxes, labels):
      x, y, w, h = box.tolist()
      class_label = int(class_label)

      i, j = int(self.S * y), int(self.S * x)
      x_cell, y_cell = self.S * x - j, self.S * y - i
      width_cell, height_cell = w * self.S, h * self.S
      
      if label_matrix[i, j, 20] == 0:
        label_matrix[i, j, 20] = 1
        label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        label_matrix[i, j, class_label] = 1

    return image, label_matrix

  