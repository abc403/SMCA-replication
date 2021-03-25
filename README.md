# Usage
There are no extra compiled components in SMCA DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/facebookresearch/detr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train Single Scale SMCA on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco --batch_size 2 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type3 --output_dir smca_single_scale


```
A single epoch takes 30 minutes, so 50 epoch training
takes around 25 hours on a single machine with 8 V100 cards.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>box AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>50</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR</td>
      <td>R50</td>
      <td>108</td>
      <td>42.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R50</td>
      <td>250</td>
      <td>43.5</td>
    </tr>
  </tbody>
</table>
