import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets2 import load_data
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

# 这里示例只写 gpu_ids=[0]，如果多卡需要相应修改
gpu_ids = [4]
torch.cuda.set_device(gpu_ids[rank])

def create_argparser():
    """
    根据 YAML 配置文件+命令行参数构建解析器。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    # 可通过命令行指定 config 路径；此处直接读取固定路径
    with open('/date/NMI/code/config/config_test.yaml', "r") as file:
        config = yaml.safe_load(file)
    add_dict_to_argparser(parser, config)
    return parser

def main():
    # 利用当前脚本名来确定输出目录前缀
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 解析参数
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    args.all_gene = 60  # 总基因数
    args.gene_num = 20  # 每组预测20个基因
    args.batch_size = 5
    logger.configure()

    logger.log("loading data...")
    # 指定模型路径，根据实际情况修改
    args.model_path = '/date/NMI/code/logs_new/x10_Xenium_g20_gene_0-59_Unet_0313-0702/model029000.pt'
    print(args)

    # 设置输出目录
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    results_dir = os.path.join(root_dir, "results", script_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logs_dir = results_dir
    progress_csv = os.path.join(logs_dir, f"{script_name}_progress.csv")
    if not os.path.exists(progress_csv):
        with open(progress_csv, mode="w") as file:
            file.write("Batch,RMSE,SSIM,CC\n")
    output_dir = os.path.join(logs_dir, f"{script_name}_temp")
    create_output_dir(output_dir)
    print("Output directory:", output_dir)

    # 根据数据集类型计算总组数：
    # 例如：对于 Xenium，原始样本数为84，每个样本拆分为 (all_gene/gene_num) 组（60/20=3）
    if args.dataset_use == 'Xenium':
        data_num = 4 * (args.all_gene // args.gene_num)
    elif args.dataset_use == 'SGE':
        data_num = 47 * (args.all_gene // args.gene_num)
    else:
        data_num = 223 * (args.all_gene // args.gene_num)

    # 加载数据生成器（注意返回的是已分组的数据，每组包含20个基因）
    data = load_superres_data(
        args.batch_size,
        args.data_root,
        args.dataset_use,
        status='Test',
        SR_times=args.SR_times,
        gene_num=args.gene_num,
        all_gene=args.all_gene
    )

    # 创建模型与扩散对象
    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(args)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating samples...")

    # 用于保存所有分组的预测结果和对应的 ground truth（GT）
    # 每个元素的形状均为 [H, W, gene_num]，即单个分组（20个基因）
    pred_group_list = []
    gt_group_list = []

    # 每个 batch 推理
    num_batches = data_num // args.batch_size
    for i in range(num_batches):
        # 根据数据集类型获取 HR 和 model_kwargs
        if args.dataset_use == 'Xenium':
            hr, model_kwargs = next(data)
            # 若 SR_times==5，则插值到 (256,256)
            if args.SR_times == 5:
                hr = F.interpolate(hr, size=(256, 256))
            # 将 HR 数据转为 numpy，形状为 [batch, H, W, gene_num]
            hr = hr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        elif args.dataset_use in ['SGE', 'BreastST']:
            model_kwargs = next(data)
            low_res_data = model_kwargs['low_res']
            model_kwargs['low_res'] = F.interpolate(low_res_data, size=(26, 26))
            hr_tensor = model_kwargs['low_res']
            print("HR shape:", hr_tensor.shape)
            hr = hr_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        # 拷贝到当前设备
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # 选择扩散采样函数
        if args.sampling_method == 'ddim':
            sample_fn = diffusion.ddim_sample_loop
        elif args.sampling_method == 'dpm++':
            sample_fn = diffusion.dpm_solver_sample_loop
        else:
            sample_fn = diffusion.p_sample_loop

        # 生成预测，输出形状为 [batch, gene_num, H, W]
        sample = sample_fn(
            model,
            (args.batch_size, args.gene_num, model_kwargs['WSI_5120'].shape[2], model_kwargs['WSI_5120'].shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # 转换为 numpy，转置为 [batch, H, W, gene_num]
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()

        # 遍历当前 batch 内的每个 sample
        for j in range(hr.shape[0]):
            gt_sample = hr[j]       # [H, W, gene_num]
            pred_sample = sample[j] # [H, W, gene_num]

            # 对预测结果进行归一化
            pred_sample = normalize_prediction(pred_sample)

            # 保存当前分组的图像（可选）
            save_sample_images(
                gt_sample, 
                pred_sample,
                batch_index=i+1,
                sample_index=j+1,
                output_dir=output_dir
            )

            # 保存当前分组结果到列表，用于后续合并
            pred_group_list.append(pred_sample)
            gt_group_list.append(gt_sample)

        # 可选：计算当前 batch 内各分组指标（此处仅针对每个20基因组）
        rmse_batch, ssim_batch, cc_batch = 0, 0, 0
        for j in range(hr.shape[0]):
            r, s, c = compute_metrics(hr[j], normalize_prediction(sample[j]))
            rmse_batch += r
            ssim_batch += s
            cc_batch += c
        batch_size_actual = hr.shape[0]
        rmse_avg = rmse_batch / batch_size_actual
        ssim_avg = ssim_batch / batch_size_actual
        cc_avg = cc_batch / batch_size_actual
        content_line = f"{i+1},{rmse_avg},{ssim_avg},{cc_avg}"
        save_metrics_to_csv(progress_csv, content_line)
        print(f"Batch {i+1} completed: RMSE={rmse_avg}, SSIM={ssim_avg}, CC={cc_avg}")

    # 推理完毕后，进行后续处理：
    # 合并各分组预测和 GT 得到完整 60 个基因的结果
    total_groups = len(pred_group_list)  # 实际总组数，例如 250
    groups_per_sample = args.all_gene // args.gene_num  # 例如 3
    num_full_groups = (total_groups // groups_per_sample) * groups_per_sample  # 83 * 3 = 249

    total_groups = len(pred_group_list)  # 实际总组数，例如 250
    groups_per_sample = args.all_gene // args.gene_num  # 例如 3
    num_full_groups = (total_groups // groups_per_sample) * groups_per_sample  # 83 * 3 = 249

    pred_group_array = np.array(pred_group_list[:num_full_groups])
    gt_group_array = np.array(gt_group_list[:num_full_groups])
    num_samples = num_full_groups // groups_per_sample  # 应该为 83

    # 然后 reshape
    pred_group_array = pred_group_array.reshape(num_samples, groups_per_sample,
                                                pred_group_array.shape[1],
                                                pred_group_array.shape[2],
                                                pred_group_array.shape[3])
    gt_group_array = gt_group_array.reshape(num_samples, groups_per_sample,
                                            gt_group_array.shape[1],
                                            gt_group_array.shape[2],
                                            gt_group_array.shape[3])

    # 在最后一个维度上拼接各组，得到完整预测：形状为 [num_samples, H, W, groups_per_sample * gene_num]，即 [num_samples, H, W, 60]
    final_pred = np.concatenate([pred_group_array[:, i, :, :, :] for i in range(groups_per_sample)], axis=-1)
    final_gt = np.concatenate([gt_group_array[:, i, :, :, :] for i in range(groups_per_sample)], axis=-1)

    print("Final merged predictions shape:", final_pred.shape)
    print("Final merged GT shape:", final_gt.shape)

    # 计算每个样本的整体指标（RMSE、SSIM、CC）
    overall_rmse_list, overall_ssim_list, overall_cc_list = [], [], []
    for idx in range(num_samples):
        r, s, c = compute_metrics(final_gt[idx], final_pred[idx])
        overall_rmse_list.append(r)
        overall_ssim_list.append(s)
        overall_cc_list.append(c)
    avg_rmse = np.mean(overall_rmse_list)
    avg_ssim = np.mean(overall_ssim_list)
    avg_cc = np.mean(overall_cc_list)
    print("Overall merged predictions metrics:")
    print("RMSE:", avg_rmse)
    print("SSIM:", avg_ssim)
    print("CC:", avg_cc)

    overall_line = f"Overall Merged,,{avg_rmse},{avg_ssim},{avg_cc}"
    save_metrics_to_csv(progress_csv, overall_line)

    # 保存每个样本完整预测的图像，每个基因单独保存
    for idx in range(num_samples):
        for gene_idx in range(final_pred.shape[-1]):
            img_save_path = os.path.join(output_dir, f"merged_sample_{idx+1}_gene_{gene_idx+1}.png")
            plt.imsave(img_save_path, final_pred[idx, :, :, gene_idx], cmap='viridis')

    print("Post-processing completed. Merged predictions and metrics saved.")

def remove_all_file(path: str):
    """
    移除指定文件夹内所有文件，不删除子文件夹。
    """
    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)

def save_metrics_to_csv(file_path: str, content: str):
    """
    追加写入指标内容到指定 CSV 文件中。
    """
    with open(file_path, mode="a") as f:
        f.write(content + "\n")

def create_output_dir(dir_path: str):
    """
    如果目录存在，则清空；否则创建该目录。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        remove_all_file(dir_path)

def save_sample_images(gt_sample: np.ndarray, pred_sample: np.ndarray, batch_index: int, sample_index: int, output_dir: str):
    """
    保存 GT 和预测结果为 PNG 图像，包含对各个基因通道的单独保存。
    """
    for gene_idx in range(gt_sample.shape[-1]):
        gt_gene_path = os.path.join(output_dir, f"sample_{batch_index}_gt_{sample_index}_gene_{gene_idx+1}.png")
        pred_gene_path = os.path.join(output_dir, f"sample_{batch_index}_pred_{sample_index}_gene_{gene_idx+1}.png")
        plt.imsave(gt_gene_path, gt_sample[..., gene_idx], cmap='viridis')
        plt.imsave(pred_gene_path, pred_sample[..., gene_idx], cmap='viridis')

def compute_metrics(gt_sample: np.ndarray, pred_sample: np.ndarray):
    """
    计算给定 GT 和 Pred 的所有基因通道下的 RMSE、SSIM、CC，并返回各通道平均值。
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
    对预测结果 pred_sample 进行智能裁剪和归一化，按通道去除最低 10% 灰度值（噪声），
    然后对每个通道裁剪到有效范围并归一化到 [0,1] 区间。
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

def load_superres_data(batch_size, data_root, dataset_use, status, SR_times, gene_num, all_gene):
    """
    加载并返回一个生成器，每次 yield 模型需要的内容。
    """
    dataset = load_data(
        data_root=data_root,
        dataset_use=dataset_use,
        status=status,
        SR_times=SR_times,
        gene_num=gene_num,
        all_gene=all_gene
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    if dataset_use == 'Xenium':
        for SR_ST, spot_ST, WSI_5120, WSI_320, gene_class, Gene_index_map in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "WSI_320": WSI_320,
                "gene_class": gene_class,
                "Gene_index_map": Gene_index_map
            }
            yield SR_ST, model_kwargs
    elif dataset_use in ['SGE', 'BreastST']:
        for spot_ST, WSI_5120, WSI_320 in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "WSI_320": WSI_320
            }
            yield model_kwargs

if __name__ == "__main__":
    main()
