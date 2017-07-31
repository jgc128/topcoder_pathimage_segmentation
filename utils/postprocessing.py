import numpy as np
import cv2


def morph_masks(masks, kernel_size=5, operation='open'):
    if operation == 'open':
        op = cv2.MORPH_OPEN
    elif operation == 'close':
        op = cv2.MORPH_CLOSE
    else:
        raise ValueError(f'Operation {operation} is unknown')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if len(masks.shape) == 3:
        masks = np.stack([cv2.morphologyEx(m, op, kernel) for m in masks], axis=0)
    else:
        masks = cv2.morphologyEx(masks, op, kernel)

    return masks
