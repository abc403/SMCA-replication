# Usage
There are no extra compiled components in DETR and package dependencies are minimal,
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
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
A single epoch takes 28 minutes, so 300 epoch training
takes around 6 days on a single machine with 8 V100 cards.
To ease reproduction of our results we provide
[results and training logs](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)
for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales an crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
We provide results for all DETR detection models in this
[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).
Note that numbers vary depending on batch size (number of images) per GPU.
Non-DC5 models were trained with batch size 2, and DC5 with 1,
so DC5 models show a significant drop in AP if evaluated with more
than 1 image per GPU.

We provide baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">download</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">download</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">download</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">download</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>


# fovea_scale_head_query_mixture
Setting
4GPU Batch 4



srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type1 --output_dir focal_300_type1_no_scale

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type3 --output_dir focal_300_type3_xy_scale

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type4 --output_dir focal_300_type4_covarince_scale

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type2 --head_mixture --output_dir focal_300_type2_scale_head_mixture

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type2 --query_mixture --output_dir focal_300_type2_scale_query_mixture

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type2 --smooth 4 --output_dir focal_300_type2_single_scale_smooth_4

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 40 --num_queries 300 --epochs 50 --dynamic_scale type2 --smooth 16 --output_dir focal_300_type2_single_scale_smooth_16

Setting
8GPU Batch 4


srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 100 --num_queries 300 --epochs 150 --dynamic_scale type2 --query_mixture --output_dir focal_300_type2_scale_query_mixture_150

srun --gres gpu:4 -c 32 -p vc_research python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ../../dataset/coco/ 
--batch_size 4 --lr_drop 400 --num_queries 300 --epochs 500 --dynamic_scale type2 --query_mixture --output_dir focal_300_type2_scale_query_mixture_500
