# PandaSet utils

Simple scripts to deal with PandaSet and method predictions.

#### Requirements

Required packages are listed in *requirements.txt* file. To install them, use ```pip3 install -r requirements.txt```

### Dataloader

Visualize PandaSet point clouds with corresponding labels.

```shell script
python3 dataloader.py <pandaset_path>
```

### Visualizer

Visualize predictions on Pandaset assuming predictions are saved as ```XX.label```. In other case as per Cylinder3D outputs, modify lines 34-35.

```shell script
python3 visualizer.py <sequence> <pandaset_path> <prediction_path>
```

### Evaluation

Evaluate predictions for PandaSet using Accuracy and mIoU. As for the visualier, you might modify the predictions format.

```shell script
python3 evaluate.py <pandaset_path> <prediction_path>
```

### Cloud Integration

Integrate clouds from a sequence. Saves the output point cloud as ```.ply``` format.

```shell script
python3 evaluate.py <pandaset_path> <output_path_ply>
```