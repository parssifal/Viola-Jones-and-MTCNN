from google.colab import drive
from typing import Callable
from pathlib import Path
from sklearn import datasets, metrics, model_selection, svm
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
%matplotlib inline

drive.mount('/content/drive')

DATASET_PATH = Path(r'/content/drive/MyDrive/ColabNotebooks/data/images')

labels_path = Path("/content/drive/MyDrive/ColabNotebooks/data/dataset_parameters.txt")

with open(labels_path) as f:
  lines = f.read().splitlines()

labels_dict = {}
for idx, l in enumerate(lines):
  values = l.split(".") 	# to detect the lines with image names  
  if values[-1] == 'jpg':
    img_name = Path(l).stem
    bbox_number = int(lines[idx + 1])
    bboxes = []
    for i in range(bbox_number):
      labels = lines[idx + 2 + i]
      x, y, w, h = [int(s) for s in labels.split(" ")[:4]]
      bboxes.append((x, y, w, h))
    labels_dict[img_name] = bboxes

def intersection_over_union(x1, y1, width1, height1, x2, y2, width2, height2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + width1 - 1, x2 + width2 - 1)
    yB = min(y1 + height1 - 1, y2 + height2 - 1)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and true rectangles
    box1Area = width1 * height1
    box2Area = width2 * height2
    #compute the intersection over union
    IoU = interArea / float(box1Area + box2Area - interArea)
    return IoU

def evaluate_detector(x, y, z, detector: Callable, dataset_path: Path, labels_dict: dict):
  r"""
  Args: 
    detector: Function to find BB on RGB image. Expected to return LIST of BB like cv2.CascadeClassifier
    dataset_path: Path to dataset with images.
  """
  files = list(DATASET_PATH.rglob("*.jpg"))
  my_time = []
  quan = 0
  X = x
  Y = y
  Z = z
  TP, TN, FP, FN = 0, 0, 0, 0
  for file in tqdm(files):
    tp, tn, fp, fn = 0, 0, 0, 0
    img_name = file.stem
    gt_bboxes = labels_dict[img_name]

    # Read file with image
    img = cv2.imread(str(file))
    img = img[:,:, ::-1] # BGR -> RGB

    # Detect faces
    pred_bboxes, my_time, quan = detector(img, my_time, quan, X, Y, Z)

    # Count per-image metrics
    #TP and FN
    for gt_bbox in gt_bboxes:
      FR = False # finding rate
      x_gt, y_gt, w_gt, h_gt = gt_bbox
      max_iou = 0
      num = 0
      i = 0
      for bbox in pred_bboxes:
        x, y, w, h = bbox
        i += 1
        IoU = intersection_over_union(x, y, w, h, x_gt, y_gt, w_gt, h_gt)
        if IoU > 0.5:
                FR = True
                if IoU > max_iou:
                    max_iou = IoU
                    num = i
        if not FR:
            fn += 1
        if FR:
            tp += 1
            pred_bboxes[num - 1] = [0, 0, 0, 0]

    # TN
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tn += 1

    #FP
    for bbox in pred_bboxes:
        if np.all(bbox != [0, 0, 0, 0]):
            fp += 1

    TP += tp
    TN += tn
    FP += fp
    FN += fn

  return TP, TN, FP, FN, my_time, quan

haarcascade_path = "/content/drive/MyDrive/ColabNotebooks/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def cascade_detector(image, my_time, i, x, y, z):
  # Convert from RGB to BGR and then to Grey
  # image = image[:,:, ::-1] # BGR <-> RGB
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  # Detect faces
  start_time = time.time()
  faces = face_cascade.detectMultiScale(gray, scaleFactor=x, minNeighbors=y, minSize=(z, z))
  end_time = time.time()
  my_time.append(end_time - start_time)
  i += 1
  return faces, my_time, i

df = pd.DataFrame(columns = ['Scale factor', 'minNeighbors', 'minSize', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'True Positive Rate', 'False Positive Rate', 'Average time', 'Standard deviation', 'Number of all images'])
# Номер 1
n = 183
x = 14.70
y = 1
for z in range(11, 30, 3): # z is changing of minSize
  TP, TN, FP, FN, my_time, i = evaluate_detector(x, y, z, cascade_detector, dataset_path=DATASET_PATH, labels_dict=labels_dict)
  AC = float(TP + TN)/float(TP + TN + FP + FN) * 100 # Accuracy
  PR = float(TP)/float(TP + FP) * 100 # Precision
  RC = float(TP)/float(TP + FN) * 100 # Recall
  F1 = float(2 * PR * RC) / float(PR + RC) # F1-score
  FPR = float(FP) / float(FP + TN) # False Positive Rate
  TPR = float(TP) / float(TP + FN) # True Positive Rate
  df.loc[n] = [x, y, z, TP, TN, FP, FN, AC, PR, RC, F1, TPR, FPR, np.mean(my_time), np.std(my_time), i]
  with pd.ExcelWriter('/content/drive/MyDrive/ColabNotebooks/data/output_data1.xlsx') as writer:
    df.to_excel(writer)
  print(n)
  n = n + 1

df = pd.DataFrame(columns = ['Scale factor', 'minNeighbors', 'minSize', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'True Positive Rate', 'False Positive Rate', 'Average time', 'Standard deviation', 'Number of all images'])
# Номер 6
n = 218
x = 16.80
y = 1
for z in range(32, 51, 3): # z is changing of minSize
  TP, TN, FP, FN, my_time, i = evaluate_detector(x, y, z, cascade_detector, dataset_path=DATASET_PATH, labels_dict=labels_dict)
  AC = float(TP + TN)/float(TP + TN + FP + FN) * 100 # Accuracy
  PR = float(TP)/float(TP + FP) * 100 # Precision
  RC = float(TP)/float(TP + FN) * 100 # Recall
  F1 = float(2 * PR * RC) / float(PR + RC) # F1-score
  FPR = float(FP) / float(FP + TN) # False Positive Rate
  TPR = float(TP) / float(TP + FN) # True Positive Rate
  df.loc[n] = [x, y, z, TP, TN, FP, FN, AC, PR, RC, F1, TPR, FPR, np.mean(my_time), np.std(my_time), i]
  with pd.ExcelWriter('/content/drive/MyDrive/ColabNotebooks/data/output_data6.xlsx') as writer:
    df.to_excel(writer)
  print(n)
  n = n + 1

df = pd.DataFrame(columns = ['Scale factor', 'minNeighbors', 'minSize', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'True Positive Rate', 'False Positive Rate', 'Average time', 'Standard deviation', 'Number of all images'])
# Номер 11
n = 253
x = 19.95
y = 1
for z in range(11, 30, 3): # z is changing of minSize
  TP, TN, FP, FN, my_time, i = evaluate_detector(x, y, z, cascade_detector, dataset_path=DATASET_PATH, labels_dict=labels_dict)
  AC = float(TP + TN)/float(TP + TN + FP + FN) * 100 # Accuracy
  PR = float(TP)/float(TP + FP) * 100 # Precision
  RC = float(TP)/float(TP + FN) * 100 # Recall
  F1 = float(2 * PR * RC) / float(PR + RC) # F1-score
  FPR = float(FP) / float(FP + TN) # False Positive Rate
  TPR = float(TP) / float(TP + FN) # True Positive Rate
  df.loc[n] = [x, y, z, TP, TN, FP, FN, AC, PR, RC, F1, TPR, FPR, np.mean(my_time), np.std(my_time), i]
  with pd.ExcelWriter('/content/drive/MyDrive/ColabNotebooks/data/output_data11.xlsx') as writer:
    df.to_excel(writer)
  print(n)
  n = n + 1
