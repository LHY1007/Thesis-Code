import argparse
import yaml
import os
import numpy as np
import argparse, time, random
from guided_diffusion import dist_util, logger
from guided_diffusion.imgdatasets3 import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser
)
import torch
from guided_diffusion.train_util import TrainLoop
from mpi4py import MPI
comm =MPI.COMM_WORLD
os.environ["TOKENIZERS_PARALLELISM"] = "false"
rank = comm.Get_rank()

# 定义 GPU ID 列表，根据 rank 来选择对应的 GPU
gpu_ids = [6]  # GPU 3 和 GPU 4
torch.cuda.set_device(gpu_ids[rank])

def main():
    # 解析命令行参数和加载配置
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    # 修改部分参数
    args.all_gene = 60  # change
    args.gene_num = 20  # change
    args.batch_size = 10  # change
    args.SR_times = 10
    args.dataset_use = 'Xenium5k'
    # 新增：基因名字文件路径（txt文件，每行一个基因名）
    #gene_order_path = os.path.join(args.data_root, 'Xenium5k_human', 'gene_order1.npy')
    n = 3
    # n=xy1,2[01] zc 34[23]  xx 56[45]  (注：此处说明可根据需要调整)
    
    for i in range(args.all_gene // args.gene_num):
        i=i+2
        loop_start_time = time.time()
        gene_order = np.load('/date/NMI/data/brain glioblastomamultiforme/Xenium5k/gene_order.npy')[(n - 1) * args.all_gene + (i * args.gene_num) : (n - 1) * args.all_gene + ((i + 1) * args.gene_num)]
        #gene_name_order = np.loadtxt('/date/NMI/data/Xenium_pancreascancer/gene_order_name.txt', dtype=str)[(n - 1) * args.all_gene + (i * args.gene_num) : (n - 1) * args.all_gene + ((i + 1) * args.gene_num)]
        cur_time = time.strftime('%m%d-%H%M', time.localtime())
        save_dir = 'logs5Kbraincancer/' + args.dataset_use + '_' + str(args.SR_times) + 'X' + '_G' + \
                   str((n - 1) * args.all_gene + (i * args.gene_num)) + '-' + \
                   str((n - 1) * args.all_gene + ((i + 1) * args.gene_num))
        save_dir = save_dir + '_{}'.format(cur_time)
        logger.configure(dir=save_dir + '/')
    
        logger.log("creating data loader...")
        # 调用加载数据函数，同时传入 gene_name_order 参数
        brain_dataset = load_superres_data(
            data_root=args.data_root,
            dataset_use=args.dataset_use,
            status='Train',
            SR_times=args.SR_times,
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            gene_order=gene_order,
        )
    
        logger.log("creating model...")
        model, diffusion = sr_create_model_and_diffusion(args)
        model.to(dist_util.dev())
    
        # 创建 schedule sampler
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
        logger.log("training...")
        # 启动训练循环
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=brain_dataset,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            SR_times=args.SR_times,
        ).run_loop()
    
        # 计算并记录循环耗时
        loop_duration = time.time() - loop_start_time
        hours = int(loop_duration // 3600)
        minutes = int((loop_duration % 3600) // 60)
        logger.log(f"循环 {i} 完成，耗时: {hours}小时{minutes}分钟")

def load_superres_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order):
    # Load the super-resolution data using the specified directories
    return load_data(data_root=data_root,dataset_use=dataset_use,status=status,SR_times=SR_times,gene_num=gene_num,all_gene=all_gene,gene_order=gene_order)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('/date/NMI/code/config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()

