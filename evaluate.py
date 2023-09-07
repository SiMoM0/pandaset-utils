# Pandaset prediction evaluator
# Usage: python3 evaluate.py <pandaset_path> <prediction_path>

import os
import sys
import gzip
import pickle
import yaml
import numpy as np

# label mapping from pandaset to kitti format
with open('./config/label_mapping.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

# learning map for labels
with open('./config/learing_map.yaml', 'r') as f:
    learning_map = yaml.safe_load(f)

n_classes = 20

# args
pandaset_path = sys.argv[1]
prediction_path = sys.argv[2]

miou = []
accuracy = []
prec = []
rec = []
f1_score = []

# loop sequences
for sequence in sorted(os.listdir(prediction_path)):

    # check if semseg folder exists
    semseg_path = os.path.join(pandaset_path, sequence, 'annotations', 'semseg')
    if not os.path.exists(semseg_path):
        continue

    print(f'Evaluating sequence ({sequence})\t', end="")

    # set of labels that appear in ground truth
    unique_gt = set()

    # confusion matrix for the current sequence
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    # accuracy for current sequence
    acc = 0.0

    for i in range(0, 80):
        #pred_path = os.path.join(prediction_path, sequence, '{0:06d}.label'.format(i)) # predictions (cylinder format)
        pred_path = os.path.join(prediction_path, sequence, '{0:02d}.label'.format(i)) # predictions (pvkd format)
        pred = np.fromfile(pred_path, dtype=np.uint32).reshape(-1, 1)
        pred = pred & 0xFFFF
        pred = np.vectorize(learning_map.__getitem__)(pred)

        label_path = os.path.join(semseg_path, '{0:02d}.pkl.gz'.format(i)) # pandaset label
        with gzip.open(label_path, 'rb') as f:
            labels = pickle.load(f)
            labels = labels.to_numpy(dtype=np.uint32)[:pred.shape[0]]
            labels = labels & 0xFFFF
            labels = np.array([label_map[label[0]] for label in labels]).reshape(-1, 1)
            labels = np.vectorize(learning_map.__getitem__)(labels)

        assert pred.shape == labels.shape

        unique_gt |= set(np.unique(labels))
        #print(unique_gt)

        idxs = tuple(np.stack((pred, labels), axis=0))
        np.add.at(conf_matrix, idxs, 1)

        correct = (labels == pred)
        acc += correct.sum() / len(correct)

    # array of true-false
    label_presence = [index in unique_gt for index in range(0, 20)]

    # clean stats
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=1) - tp
    fn = conf_matrix.sum(axis=0) - tp

    intersection = tp
    union = tp + fp + fn + 1e-15

    iou = intersection / union
    iou_mean = (intersection[label_presence] / union[label_presence]).mean()

    #precision = (tp[label_presence] / (tp[label_presence] + fp[label_presence])).mean()
    #recall = (tp[label_presence] / (tp[label_presence] + fn[label_presence])).mean()
    #f1 = 2 * precision * recall / (precision + recall)

    #print('IOU ', iou)
    #print('MEAN IOU: ', iou_mean)    

    # semantic-kitti-api approach
    #class_iou = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) -np.diag(conf_matrix))
    #print(class_iou)

    #curr_miou = compute_miou(class_iou, label_presence)

    #print(f'Acc = {acc / 80} | mIoU = {iou_mean} | precision = {precision} | recall = {recall} | f1 = {f1}')
    print(f'Acc = {acc / 80} | mIoU = {iou_mean}')

    miou.append(iou_mean)
    accuracy.append(acc / 80)
    #prec.append(precision)
    #rec.append(recall)
    #f1_score.append(f1)

val_miou = np.mean(miou)

print('*'*40)
print(f'Accuracy = {np.mean(accuracy)}')
print(f'mIoU = {val_miou}')
#print(f'Precision = {np.mean(prec)}')
#print(f'Recall = {np.mean(rec)}')
#print(f'F1 = {np.mean(f1_score)}')
print('*'*40)