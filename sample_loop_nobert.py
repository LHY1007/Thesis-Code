import os
import re
import glob
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
# 使用与训练相同的数据加载接口，支持 gene_name_order 参数
from guided_diffusion.imgdatasets3 import load_data  
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
)
import scipy
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# GPU设置（根据需要，可修改为与训练端一致）
gpu_ids = [1]  
torch.cuda.set_device(gpu_ids[rank])

def create_argparser():
    """参数解析器：可根据需要选择使用测试配置或训练配置"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    
    # 这里可根据实际情况选择 config 文件路径
    with open('/date/NMI/code/config/config_test.yaml', "r") as file:
        config = yaml.safe_load(file)
    add_dict_to_argparser(parser, config)
    return parser

def main():
    # 解析参数并初始化分布式设置
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    
    # 参数设置保持与训练代码一致
    args.all_gene = 20
    args.gene_num = 10
    args.batch_size = 1
    args.SR_times = 10
    args.dataset_use = 'Xenium5k'
    
    # 文件路径统一采用训练时的路径（根据实际情况修改）
    gene_order_path = '/date/NMI/data/brain glioblastomamultiforme/Xenium5k/gene_order.npy'
    #gene_name_order_path = '/date/NMI/data/Xenium_pancreascancer/gene_order_name.txt'
    
    # 查找所有符合条件的训练模型目录（注意：训练时日志保存在 logs5K1/ 目录下）
    model_dirs = glob.glob(os.path.join("logs5Kbraincancer1", f"{args.dataset_use}_{args.SR_times}X_G*"))
    print(f"查找到的模型目录前缀: {args.dataset_use}_{args.SR_times}X_G")
    
    for model_dir in model_dirs:
        # 从目录名中解析基因范围信息
        print("Processing:", model_dir)
        dir_name = os.path.basename(model_dir)
        g_part = dir_name.split("_G")[1].split("_")[0]
        start_gene = int(g_part.split("-")[0])
        print(f"处理基因组起始索引: {start_gene}")
        
        # 加载对应的基因顺序和基因名称顺序
        gene_order = np.load(gene_order_path)[start_gene:start_gene+args.gene_num]
        #gene_name_order = np.loadtxt(gene_name_order_path, dtype=str)[start_gene:start_gene+args.gene_num]
        
        # 查找最新模型参数
        checkpoints = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not checkpoints:
            print(f"跳过无检查点的目录: {model_dir}")
            continue
            
        max_step = max(
            [int(re.search(r"model(\d+)\.pt", ckpt).group(1)) 
             for ckpt in checkpoints if re.search(r"model\d+\.pt", ckpt)]
        )
        model_path = os.path.join(model_dir, f"model{max_step:06d}.pt")

        # 准备结果目录，目录名称中保持与训练时一致的命名逻辑
        script_name = f'Ours-{args.dataset_use}/{args.SR_times}X/G{start_gene}-{start_gene+args.gene_num}'
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        results_dir = os.path.join(root_dir, "resultsbraincancer1", script_name)
        os.makedirs(results_dir, exist_ok=True)
        logger.configure(dir=results_dir)

        # 初始化模型
        logger.log(f"\n{'='*40}\n测试基因组 {start_gene}-{start_gene+args.gene_num}\n{'='*40}")
        model, diffusion = sr_create_model_and_diffusion(args)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        # 初始化数据集，调用修改后的 load_superres_data（传入 gene_order 和 gene_name_order）
        logger.log("loading data...")
        # 这里根据 dataset_use 选择数据数量
        data_num = 14 if args.dataset_use == 'Xenium' else 14
        
        data = load_superres_data(
            args.batch_size,
            data_root=args.data_root,
            dataset_use=args.dataset_use,
            status='Test',
            SR_times=args.SR_times,
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            gene_order=gene_order,
        )

        # 初始化结果记录（CSV）
        progress_csv = os.path.join(results_dir, "metrics.csv")
        with open(progress_csv, "w") as f:
            f.write("SampleID,RMSE,SSIM,CC\n")
        
        output_dir = os.path.join(results_dir, "samples")
        create_output_dir(output_dir)

        # 推理流程
        logger.log("creating samples...")
        rmse_all, ssim_all, cc_all = 0, 0, 0
        num_batches = data_num // args.batch_size

        for batch_idx in range(num_batches):
            # 数据预处理
            if args.dataset_use in ['Xenium', 'Xenium5k']:
                hr, model_kwargs = next(data)
                if args.SR_times == 5:
                    hr = F.interpolate(hr, size=(256, 256))
                hr = hr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            elif args.dataset_use in ['SGE', 'BreastST']:
                model_kwargs = next(data)
                low_res_data = model_kwargs['low_res']
                model_kwargs['low_res'] = F.interpolate(low_res_data, size=(26, 26))
                hr_tensor = model_kwargs['low_res']
                hr = hr_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            # 将数据移动到设备
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            
            # 选择采样函数
            if args.sampling_method == 'ddim':
                sample_fn = diffusion.ddim_sample_loop
            elif args.sampling_method == 'dpm++':
                sample_fn = diffusion.dpm_solver_sample_loop
            else:
                sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (
                    args.batch_size,
                    args.gene_num,
                    model_kwargs['WSI_5120'].shape[2],
                    model_kwargs['WSI_5120'].shape[3]
                ),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = sample.permute(0, 2, 3, 1).cpu().numpy()

            # 指标计算
            rmse_batch, ssim_batch, cc_batch = 0, 0, 0
            for j in range(hr.shape[0]):
                gt_sample = hr[j]
                pred_sample = sample[j]

                pred_sample_t = torch.tensor(pred_sample).permute(2, 0, 1).unsqueeze(0)
                pred_sample_t = F.interpolate(pred_sample_t, size=(128, 128))
                pred_sample_t = F.interpolate(pred_sample_t, size=(256, 256))
                pred_sample = pred_sample_t.squeeze(0).permute(1, 2, 0).numpy()
                pred_sample = normalize_prediction(pred_sample)

                # 保存示例图像
                save_sample_images(
                    gt_sample, 
                    pred_sample,
                    batch_index=batch_idx+1,
                    sample_index=j+1,
                    output_dir=output_dir
                )

                # 计算指标
                rmse_val, ssim_val, cc_val = compute_metrics(gt_sample, pred_sample)
                rmse_batch += rmse_val
                ssim_batch += ssim_val
                cc_batch += cc_val

            batch_size_actual = hr.shape[0]
            rmse_avg = rmse_batch / batch_size_actual
            ssim_avg = ssim_batch / batch_size_actual
            cc_avg = cc_batch / batch_size_actual

            rmse_all += rmse_avg
            ssim_all += ssim_avg
            cc_all += cc_avg

            with open(progress_csv, "a") as f:
                f.write(f"{batch_idx+1},{rmse_avg},{ssim_avg},{cc_avg}\n")

            logger.log(f"批次 {batch_idx+1} 完成: RMSE={rmse_avg:.4f}, SSIM={ssim_avg:.4f}, CC={cc_avg:.4f}")

        overall_rmse = rmse_all / num_batches
        overall_ssim = ssim_all / num_batches
        overall_cc = cc_all / num_batches

        with open(progress_csv, "a") as f:
            f.write(f"Overall,,{overall_rmse},{overall_ssim},{overall_cc}\n")

        logger.log(f"基因组 {start_gene}-{start_gene+args.gene_num} 测试完成")
        logger.log(f"最终指标: RMSE={overall_rmse:.4f}, SSIM={overall_ssim:.4f}, CC={overall_cc:.4f}")

def remove_all_file(path: str):
    """
    移除指定文件夹内所有文件，不删除子文件夹。
    """
    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)

def create_output_dir(dir_path: str):
    """
    如果目录存在，则清空；否则创建该目录。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        remove_all_file(dir_path)

def save_sample_images(
    gt_sample: np.ndarray, 
    pred_sample: np.ndarray,
    batch_index: int,
    sample_index: int,
    output_dir: str
):
    """
    保存 GT 和预测结果为 PNG 图像，对每个基因通道单独保存。
    """
    for gene_idx in range(gt_sample.shape[-1]):
        gt_gene_path = os.path.join(
            output_dir, f"sample_{batch_index}_gt_{sample_index}_gene_{gene_idx+1}.png"
        )
        pred_gene_path = os.path.join(
            output_dir, f"sample_{batch_index}_pred_{sample_index}_gene_{gene_idx+1}.png"
        )
        plt.imsave(gt_gene_path, gt_sample[..., gene_idx], cmap='viridis')
        plt.imsave(pred_gene_path, pred_sample[..., gene_idx], cmap='viridis')

def compute_metrics(gt_sample: np.ndarray, pred_sample: np.ndarray):
    """
    计算每个基因通道的 RMSE、SSIM、CC，返回各通道平均值。
    """
    rmse_gene, ssim_gene, cc_gene = [], [], []
    for gene_idx in range(gt_sample.shape[-1]):
        gt_gene = gt_sample[..., gene_idx]
        pred_gene = pred_sample[..., gene_idx]
        if np.std(gt_gene) == 0:
            continue
        mse_loss = np.mean((gt_gene - pred_gene) ** 2)
        rmse_gene.append(np.sqrt(mse_loss))
        ssim_value = structural_similarity(gt_gene, pred_gene, data_range=1.0)
        ssim_gene.append(ssim_value)
        cc_value, _ = scipy.stats.pearsonr(gt_gene.flatten(), pred_gene.flatten())
        cc_gene.append(abs(cc_value))
    rmse_avg = np.mean(rmse_gene) if len(rmse_gene) > 0 else 0
    ssim_avg = np.mean(ssim_gene) if len(ssim_gene) > 0 else 0
    cc_avg = np.mean(cc_gene) if len(cc_gene) > 0 else 0
    return rmse_avg, ssim_avg, cc_avg

def normalize_prediction(pred_sample: np.ndarray):
    """
    对预测结果进行智能裁剪和归一化：按通道去除低于分位点的噪声后归一化到 [0,1]。
    """
    for k in range(pred_sample.shape[-1]):
        channel_data = pred_sample[..., k]
        lower_bound = np.percentile(channel_data, 30)
        channel_data[channel_data < lower_bound] = lower_bound
        pred_min, pred_max = np.min(channel_data), np.max(channel_data)
        if pred_max > pred_min:
            pred_sample[..., k] = (channel_data - pred_min) / (pred_max - pred_min)
        else:
            pred_sample[..., k] = 0.0
    return pred_sample

def load_superres_data(batch_size,data_root, dataset_use, status, SR_times, gene_num, all_gene, gene_order):
    """
    加载数据，返回一个生成器，每次 yield 模型所需内容，支持额外传入 gene_name_order 参数。
    """
    dataset = load_data(
        data_root=data_root,
        dataset_use=dataset_use,
        status=status,
        SR_times=SR_times,
        gene_num=gene_num,
        all_gene=all_gene,
        gene_order=gene_order,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 注意：此处使用全局参数 args.batch_size（或可传入 batch_size 参数）
        shuffle=False,
        num_workers=10,
        drop_last=False,
        pin_memory=True
    )
    
    if dataset_use in ['Xenium', 'Xenium5k']:
        for SR_ST, spot_ST, WSI_5120,WSI_320, gene_class, Gene_index_map in loader:#new1.5: gene_class, Gene_index_map
            model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "WSI_320": WSI_320,"gene_class": gene_class, "Gene_index_map": Gene_index_map}##new1.5: gene_class, Gene_index_map
            yield SR_ST, model_kwargs
    elif dataset_use in ['SGE', 'BreastST']:
        for spot_ST, WSI_5120, Gene_index_map in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "Gene_index_map": Gene_index_map
            }
            yield model_kwargs

if __name__ == "__main__":
    main()
