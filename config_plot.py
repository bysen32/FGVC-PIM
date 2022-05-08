import argparse
import os

def get_args():

    parser = argparse.ArgumentParser("Plot heatmap")
    # save path and dataset information
    parser.add_argument("--threshold", default=0.75, type=float, 
        help="heatmap threshold")

    parser.add_argument("--pretrained_path", default="./records/CUB200#SwinVit@onlyori384/backup/last.pth", type=str)
    parser.add_argument("--img_path", default="./plot_imgs/Brewer_Blackbird_0065_2310.jpg", type=str)
    parser.add_argument("--target_class", default=52, type=int)
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str, 
        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    
    # = = = = = building mode = = = = = 
    parser.add_argument("--use_fpn", default=False, type=bool)
    parser.add_argument("--use_ori", default=True, type=bool)
    parser.add_argument("--use_gcn", default=False, type=bool)
    parser.add_argument("--use_layers", 
        default=[False, False, False, False], type=list)
    parser.add_argument("--use_selections", 
        default=[False, False, False, False], type=list)
    parser.add_argument("--num_selects",
        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)
    parser.add_argument("--num_classes", default=200, type=int)

    parser.add_argument("--debug_mode", default=False, type=bool)
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    return args
