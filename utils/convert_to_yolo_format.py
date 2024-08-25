import numpy as np


def convert_to_yolo_format(target, class_mapping):

    annotations = target['annotation']['object']

    # Get the real width and height of the image from the annotation.
    real_width = int(target['annotation']['size']['width'])
    real_height = int(target['annotation']['size']['height'])

    # Ensure that annotations is a list, even if there's only one object.
    if not isinstance(annotations, list):
        annotations = [annotations]

    # Initialize an empty list to store the converted bounding boxes.
    boxes = []

    # Loop through each annotation and convert it to YOLO format.
    for anno in annotations:
        xmin = int(anno['bndbox']['xmin']) / real_width
        xmax = int(anno['bndbox']['xmax']) / real_width
        ymin = int(anno['bndbox']['ymin']) / real_height
        ymax = int(anno['bndbox']['ymax']) / real_height

        # Calculate the center coordinates, width, and height of the bounding box.
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        # Retrieve the class name from the annotation and map it to an integer ID.
        class_name = anno['name']
        class_id = class_mapping[class_name] if class_name in class_mapping else 0

        # Append the YOLO formatted bounding box to the list.
        boxes.append([class_id, x_center, y_center, width, height])

    # Convert the list of boxes to a torch tensor.
    return np.array(boxes)


  