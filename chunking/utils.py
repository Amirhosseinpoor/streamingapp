import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def contour_to_one_hot_torch(contour, num_classes, height, width):
    mask = np.zeros((height, width))
    for i in contour.keys():
        mask = cv.fillPoly(mask, [contour[i]], int(i))
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


def image_to_torch(frame: np.ndarray, device='cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device) / 255
    im_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    frame_norm = im_normalization(frame)
    return frame_norm, frame


def mask2poly(mask: np.ndarray, percentage: float, epsilon_step: float = 0.05) -> np.ndarray:
    """
    Find contours in the mask, simplify them, and return as a float32 array.

    Args:
        mask (np.ndarray): A binary mask.
        epsilon_ratio (float): A ratio to determine epsilon value for contour simplification.
            It is multiplied by the contour's perimeter to calculate epsilon.

    Returns:
        np.ndarray: Simplified contours represented as a float32 array.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour based on its area
        largest_contour = max(contours, key=cv.contourArea)
        target_points = max(int(len(largest_contour) * (percentage)), 3)
        # contour_area = cv2.contourArea(largest_contour)
        # adaptive_epsilon_ratio = base_epsilon_ratio * np.sqrt(contour_area)

        # Calculate epsilon based on the contour's perimeter
        # Epsilon is the maximum distance from the contour to the approximated contour.
        # epsilon = base_epsilon_ratio * cv2.arcLength(largest_contour, True)

        # Approximate the contour shape to simplify it
        if len(largest_contour) <= target_points:
            simplified_contour = largest_contour.reshape(-1, 2)
            return simplified_contour.astype("float32")
        epsilon = 0
        approximated_points = largest_contour
        while True:
            epsilon += epsilon_step
            new_approximated_points = cv.approxPolyDP(largest_contour, epsilon, closed=True)
            if len(new_approximated_points) > target_points:
                # print(len(new_approximated_points))
                approximated_points = new_approximated_points
            else:
                break

                # Reshape the contour array for consistency
        simplified_contour = approximated_points.reshape(-1, 2)
    else:
        simplified_contour = np.zeros((0, 2))

    return simplified_contour.astype("float32")


def torch_prob_to_contour(prob, approx):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    # print(np.unique(mask))
    contour = {}
    for k in np.unique(mask):
        if k != 0:
            contour[k] = mask2poly((mask == k).astype(np.uint8), percentage=approx).astype(np.int32)
            # print(contour[k],k)
    return contour