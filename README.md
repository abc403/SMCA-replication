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
