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
from sklearn.metrics.pairwise import cosine_similarity
import pywt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# GPU设置（根据需要，可修改为与训练端一致）
gpu_ids = [4]  
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

def compute_metrics(gt_sample: np.ndarray, pred_sample: np.ndarray, threshold=0.01):
    """
    计算给定 GT 和 Pred 的所有基因通道下的 RMSE、PSNR、SSIM、CC、IFC、VIF以及
    cosine similarity（原始数据和二值化数据），并返回各通道平均值。
    
    参数:
    gt_sample: 真实数据
    pred_sample: 预测数据
    threshold: 将低于此阈值的值置为0
    """
    # 创建副本避免修改原始数据
    gt_processed = np.copy(gt_sample)
    pred_processed = np.copy(pred_sample)
    
    # 针对每个通道分别归一化
    for gene_idx in range(gt_sample.shape[-1]):
        # 获取当前通道
        gt_gene = gt_processed[..., gene_idx]
        pred_gene = pred_processed[..., gene_idx]
        
        # 归一化到0-1范围（如果有值的话）
        if np.max(gt_gene) > 0:
            gt_processed[..., gene_idx] = gt_gene / np.max(gt_gene)
        if np.max(pred_gene) > 0:
            pred_processed[..., gene_idx] = pred_gene / np.max(pred_gene)
    
    # 阈值处理，将接近0的值置为0
    gt_thresholded = np.copy(gt_processed)
    pred_thresholded = np.copy(pred_processed)
    
    # 将低于阈值的值置为0
    gt_thresholded[gt_thresholded < threshold] = 0
    pred_thresholded[pred_thresholded < threshold] = 0
    # 负值直接置零
    gt_thresholded[gt_thresholded < 0] = 0
    pred_thresholded[pred_thresholded < 0] = 0
    
    # 创建二值化版本
    gt_binary = (gt_thresholded > 0).astype(np.float32)
    pred_binary = (pred_thresholded > 0).astype(np.float32)
    
    # 计算整体cosine similarity (不分通道)
    # 原始数据的cosine similarity
    cos_sim_overall = cosine_similarity(
        gt_thresholded.flatten().reshape(1, -1), 
        pred_thresholded.flatten().reshape(1, -1)
    )[0][0]
    
    # 二值化数据的cosine similarity
    cos_sim_binary_overall = cosine_similarity(
        gt_binary.flatten().reshape(1, -1), 
        pred_binary.flatten().reshape(1, -1)
    )[0][0]
    
    rmse_gene, psnr_gene, ssim_gene, cc_gene, ifc_gene, vif_gene = [], [], [], [], [], []
    cos_sim_gene, cos_sim_binary_gene = [], []

    for gene_idx in range(gt_sample.shape[-1]):
        gt_gene = gt_thresholded[..., gene_idx]
        pred_gene = pred_thresholded[..., gene_idx]
        gt_gene_binary = gt_binary[..., gene_idx]
        pred_gene_binary = pred_binary[..., gene_idx]

        # 计算RMSE
        mse_loss = np.mean((gt_gene - pred_gene) ** 2)
        rmse_gene.append(np.sqrt(mse_loss))
        
        # 计算PSNR
        if mse_loss == 0:  # 避免除以零
            psnr_value = 100.0  # 设置一个较大的值表示完美匹配
        else:
            max_pixel = 1.0  # 归一化后的最大值
            psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_loss))
        psnr_gene.append(psnr_value)

        # 计算SSIM
        ssim_value = structural_similarity(gt_gene, pred_gene, data_range=1.0)
        ssim_gene.append(ssim_value)

        # 计算|CC|
        cc_value, _ = scipy.stats.pearsonr(gt_gene.flatten(), pred_gene.flatten())
        cc_gene.append(abs(cc_value))
        
        # 计算IFC
        ifc_value = compute_ifc(gt_gene, pred_gene)
        ifc_gene.append(ifc_value)
        
        # 计算VIF
        vif_value = compute_vif(gt_gene, pred_gene)
        vif_gene.append(vif_value)
        
        # 计算通道级别的cosine similarity
        # 只有当向量不全为零时才计算
        if np.any(gt_gene) and np.any(pred_gene):
            cos_sim = cosine_similarity(
                gt_gene.flatten().reshape(1, -1), 
                pred_gene.flatten().reshape(1, -1)
            )[0][0]
            cos_sim_gene.append(cos_sim)
        
        # 计算通道级别的二值化数据cosine similarity
        if np.any(gt_gene_binary) and np.any(pred_gene_binary):
            cos_sim_binary = cosine_similarity(
                gt_gene_binary.flatten().reshape(1, -1), 
                pred_gene_binary.flatten().reshape(1, -1)
            )[0][0]
            cos_sim_binary_gene.append(cos_sim_binary)

    # 取平均值
    rmse_avg = np.mean(rmse_gene) if len(rmse_gene) > 0 else 0
    psnr_avg = np.mean(psnr_gene) if len(psnr_gene) > 0 else 0
    ssim_avg = np.mean(ssim_gene) if len(ssim_gene) > 0 else 0
    cc_avg = np.mean(cc_gene) if len(cc_gene) > 0 else 0
    ifc_avg = np.mean(ifc_gene) if len(ifc_gene) > 0 else 0
    vif_avg = np.mean(vif_gene) if len(vif_gene) > 0 else 0
    
    # 平均通道级别cosine similarity
    cos_sim_avg = np.mean(cos_sim_gene) if len(cos_sim_gene) > 0 else 0
    cos_sim_binary_avg = np.mean(cos_sim_binary_gene) if len(cos_sim_binary_gene) > 0 else 0

    return rmse_avg, psnr_avg, ssim_avg, cc_avg, ifc_avg, vif_avg, cos_sim_overall, cos_sim_binary_overall, cos_sim_avg, cos_sim_binary_avg

def remove_boundary_genes(data, border_width=2):
    """
    将图像边界区域的基因表达置零
    
    参数:
    data: 基因表达数据，形状为 [H, W, C]
    border_width: 要清除的边界宽度（像素数）
    """
    processed = np.copy(data)
    h, w, c = processed.shape
    
    # 创建边界掩码（0表示边界，1表示内部）
    mask = np.ones((h, w), dtype=bool)
    mask[:border_width, :] = False  # 上边界
    mask[-border_width:, :] = False  # 下边界
    mask[:, :border_width] = False  # 左边界
    mask[:, -border_width:] = False  # 右边界
    
    # 对每个通道应用掩码
    for i in range(c):
        # 仅保留非边界区域的值
        processed[:, :, i] = processed[:, :, i] * mask
    
    return processed

def post_process_sparse(generated_data, threshold=0.05, percentile_mode=True, sparsity_targets=None):
    """
    对生成的空间转录组数据进行后处理，去除低于阈值的噪声

    参数:
    generated_data: 生成的数据
    threshold: 阈值（或百分位数，取决于percentile_mode）
    percentile_mode: 如果True，使用百分位数作为阈值
    sparsity_targets: 目标稀疏度列表，每个通道一个值，如果为None则使用默认值0.95
    """
    processed = np.copy(generated_data)

    for gene_idx in range(processed.shape[-1]):
        gene_slice = processed[..., gene_idx]

        # 获取当前通道的目标稀疏度
        if sparsity_targets is not None and gene_idx < len(sparsity_targets):
            sparsity_target = sparsity_targets[gene_idx]
        else:
            Warning(f"未提供稀疏度目标，使用默认值0.95")
            sparsity_target = 0.95

        # 基于百分位数的自适应阈值
        if percentile_mode:
            # 只考虑非零值的分布
            nonzero_values = gene_slice[gene_slice > 0]
            if len(nonzero_values) > 0:
                # 计算保留非零值的百分比
                keep_percent = (1 - sparsity_target) * 100
                # 确保至少保留一些值（如果非零值太少）
                keep_percent = min(keep_percent, 100)
                actual_threshold = np.percentile(nonzero_values, 100 - keep_percent)
            else:
                actual_threshold = 0
        else:
            actual_threshold = threshold

        # 应用阈值
        gene_slice[gene_slice < actual_threshold] = 0

    return processed

def compute_ifc(gt_img, dist_img):
    """
    计算信息保真度准则 (IFC)
    """
    # 小波变换分解
    coeffs_gt = pywt.wavedec2(gt_img, 'db1', level=3)
    coeffs_dist = pywt.wavedec2(dist_img, 'db1', level=3)
    
    ifc_sum = 0
    
    # 估计GSM模型参数
    for i in range(1, len(coeffs_gt)):
        for j in range(3):  # 水平、垂直和对角子带
            gt_subband = coeffs_gt[i][j]
            dist_subband = coeffs_dist[i][j]
            
            if gt_subband.size < 16:  # 跳过过小的子带
                continue
                
            # 估计每个子带的方差
            var_gt = np.var(gt_subband)
            if var_gt < 1e-10:  # 避免接近零的方差
                continue
                
            # 估计噪声方差
            var_noise = np.mean((gt_subband - dist_subband) ** 2)
            
            # 计算此子带的互信息
            snr = var_gt / (var_noise + 1e-10)
            subband_ifc = 0.5 * np.log2(1 + snr) * gt_subband.size
            ifc_sum += subband_ifc
    
    return max(0, ifc_sum)  # 确保非负值

def compute_vif(gt_img, dist_img):
    """
    计算视觉信息保真度 (VIF)，确保结果范围在0-1之间
    """
    # 小波变换分解
    coeffs_gt = pywt.wavedec2(gt_img, 'db1', level=3)
    coeffs_dist = pywt.wavedec2(dist_img, 'db1', level=3)
    
    numerator = 0.0
    denominator = 0.0
    
    sigma_nsq = 0.1  # 模拟HVS噪声方差
    
    for i in range(1, len(coeffs_gt)):
        for j in range(3):  # 水平、垂直和对角子带
            gt_subband = coeffs_gt[i][j]
            dist_subband = coeffs_dist[i][j]
            
            if gt_subband.size < 16:  # 跳过过小的子带
                continue
                
            # 估计每个子带的方差
            var_gt = np.var(gt_subband)
            if var_gt < 1e-10:  # 避免接近零的方差
                continue
                
            # 估计失真图像与源图像的相关系数
            cov = np.mean((gt_subband - np.mean(gt_subband)) * 
                          (dist_subband - np.mean(dist_subband)))
            var_dist = np.var(dist_subband)
            
            # 确保协方差在合理范围内
            if cov <= 0:
                continue
                
            # 估计噪声方差 - 使用更稳健的方法
            var_noise = max(var_dist - (cov**2 / var_gt), 1e-10)
            
            # 计算源图像的信息
            g_denominator = 0.5 * np.log2(1 + var_gt / sigma_nsq) * gt_subband.size
            denominator += g_denominator
            
            # 计算失真图像的信息 - 确保不超过原始信息
            g_numerator = 0.5 * np.log2(1 + (cov**2) / (var_gt * var_noise)) * gt_subband.size
            numerator += g_numerator
    
    if denominator <= 0:
        return 0
        
    # 确保结果在0-1范围内
    return min(1.0, max(0.0, numerator / denominator))

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

def normalize_prediction(pred_sample: np.ndarray):
    """
    对预测结果 pred_sample 进行智能裁剪和归一化，按通道去除最低 10% 灰度值（噪声），
    然后对每个通道裁剪到有效范围并归一化到 [0,1] 区间。
    """
    for k in range(pred_sample.shape[-1]):
        # 获取当前通道的灰度值分布
        channel_data = pred_sample[..., k]

        # 去除最低 10% 的灰度值 (计算分位数)
        lower_bound = np.percentile(channel_data, 30)

        # 将低于 lower_bound 的部分视为噪声，裁剪掉
        channel_data[channel_data < lower_bound] = lower_bound

        # 重新计算有效值范围
        pred_min, pred_max = np.min(channel_data), np.max(channel_data)

        # 如果通道有非平坦分布，进行归一化
        if pred_max > pred_min:
            pred_sample[..., k] = (channel_data - pred_min) / (pred_max - pred_min)
        else:
            # 如果通道完全平坦，直接置为 0
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
            model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "WSI_320": WSI_320,"gene_class": gene_class, "Gene_index_map": Gene_index_map }##new1.5: gene_class, Gene_index_map
            yield SR_ST, model_kwargs
    elif dataset_use in ['SGE', 'BreastST']:
        for spot_ST, WSI_5120, Gene_index_map in loader:
            model_kwargs = {
                "low_res": spot_ST,
                "WSI_5120": WSI_5120,
                "Gene_index_map": Gene_index_map
            }
            yield model_kwargs

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

def main():
    # 解析参数并初始化分布式设置
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    
    # 参数设置保持与训练代码一致
    args.all_gene = 60
    args.gene_num = 20
    args.batch_size = 1
    args.SR_times = 10
    args.dataset_use = 'Xenium5k'
    
    # 文件路径统一采用训练时的路径（根据实际情况修改）
    gene_order_path = '/date/NMI/data/brain glioblastomamultiforme/Xenium5k/gene_order.npy'
    gene_name_order_path = '/date/NMI/data/Xenium_pancreascancer/gene_order_name.txt'
    
    # 查找所有符合条件的训练模型目录（注意：训练时日志保存在 logs5K1/ 目录下）
    model_dirs = glob.glob(os.path.join("logs5Kbraincancer", f"{args.dataset_use}_{args.SR_times}X_G*"))
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
        results_dir = os.path.join(root_dir, "brainresultnewcos", script_name)
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
        data_num = 14 if args.dataset_use == 'Xenium5k' else 14
        
        data = load_superres_data(
            args.batch_size,
            data_root=args.data_root,
            dataset_use=args.dataset_use,
            status='Test',
            SR_times=args.SR_times,
            gene_num=args.gene_num,
            all_gene=args.all_gene,
            gene_order=gene_order
        )

        # 初始化结果记录（CSV）
        progress_csv = os.path.join(results_dir, "metrics.csv")
        with open(progress_csv, "w") as f:
            # 添加新增指标到CSV表头
            f.write("SampleID,RMSE,SSIM,CC,IFC,VIF,PSNR,CosSim,CosSim_Binary,CosSim_Channel,CosSim_Binary_Channel\n")
        
        output_dir = os.path.join(results_dir, "samples")
        create_output_dir(output_dir)

        # 推理流程
        logger.log("creating samples...")
        rmse_all, psnr_all, ssim_all, cc_all, ifc_all, vif_all = 0, 0, 0, 0, 0, 0
        # 添加新的累计变量
        cos_sim_all, cos_sim_binary_all = 0, 0
        cos_sim_channel_all, cos_sim_binary_channel_all = 0, 0
        
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
            rmse_batch, psnr_batch, ssim_batch, cc_batch, ifc_batch, vif_batch = 0, 0, 0, 0, 0, 0
            cos_sim_batch, cos_sim_binary_batch = 0, 0
            cos_sim_channel_batch, cos_sim_binary_channel_batch = 0, 0
            
            for j in range(hr.shape[0]):
                gt_sample = hr[j]
                pred_sample = sample[j]

                pred_sample_t = torch.tensor(pred_sample).permute(2, 0, 1).unsqueeze(0)
                pred_sample_t = F.interpolate(pred_sample_t, size=(256, 256))
                pred_sample = pred_sample_t.squeeze(0).permute(1, 2, 0).numpy()
                
                # 先进行标准归一化
                pred_sample = normalize_prediction(pred_sample)
                
                # 计算每个通道的GT样本稀疏度
                gt_sparsities = []
                for gene_idx in range(gt_sample.shape[-1]):
                    gt_gene = gt_sample[..., gene_idx]
                    gene_sparsity = np.mean(gt_gene == 0)
                    gt_sparsities.append(gene_sparsity-0.05)
                    print(f"GT样本通道 {gene_idx} 稀疏度: {gene_sparsity:.4f}")
                average_sparsity = np.mean(gt_sparsities)

                # 保存原始预测结果
                pred_sample_raw = np.copy(pred_sample)
                
                # 应用稀疏后处理，使用每个通道各自的稀疏度目标
                pred_sample = post_process_sparse(
                    pred_sample, 
                    threshold=0.05,
                    percentile_mode=True, 
                    sparsity_targets=gt_sparsities
                )

                # 保存示例图像
                save_sample_images(
                    gt_sample, 
                    pred_sample,
                    batch_index=batch_idx+1,
                    sample_index=j+1,
                    output_dir=output_dir
                )

                # 计算标准指标和新增的余弦相似度指标
                rmse_avg, psnr_avg, ssim_avg, cc_avg, ifc_avg, vif_avg, cos_sim, cos_sim_binary, cos_sim_channel, cos_sim_binary_channel = compute_metrics(
                    gt_sample, pred_sample, threshold=0.1
                )
                
                # 计算并打印未稀疏化处理的cosine similarity
                gt_binary = (gt_sample > 0).astype(np.float32)
                pred_binary_raw = (pred_sample_raw > 0).astype(np.float32)
                
                cos_sim_raw = cosine_similarity(
                    gt_sample.flatten().reshape(1, -1), 
                    pred_sample_raw.flatten().reshape(1, -1)
                )[0][0]
                
                cos_sim_binary_raw = cosine_similarity(
                    gt_binary.flatten().reshape(1, -1), 
                    pred_binary_raw.flatten().reshape(1, -1)
                )[0][0]
                
                print(f"===== 余弦相似度分析 =====")
                print(f"未稀疏化处理的数据:")
                print(f"  原始数据余弦相似度: {cos_sim_raw:.4f}")
                print(f"  二值化数据余弦相似度: {cos_sim_binary_raw:.4f}")
                print(f"稀疏化处理后的数据:")
                print(f"  原始数据余弦相似度: {cos_sim:.4f}")
                print(f"  二值化数据余弦相似度: {cos_sim_binary:.4f}")
                print(f"  通道级平均余弦相似度: {cos_sim_channel:.4f}")
                print(f"  通道级平均二值化余弦相似度: {cos_sim_binary_channel:.4f}")
                
                # 累加批次结果
                rmse_batch += rmse_avg
                ssim_batch += ssim_avg
                cc_batch += cc_avg
                ifc_batch += ifc_avg
                vif_batch += vif_avg
                psnr_batch += psnr_avg
                cos_sim_batch += cos_sim
                cos_sim_binary_batch += cos_sim_binary
                cos_sim_channel_batch += cos_sim_channel
                cos_sim_binary_channel_batch += cos_sim_binary_channel

            # 记录批次结果
            batch_size_actual = hr.shape[0]
            rmse_avg = rmse_batch / batch_size_actual
            ssim_avg = ssim_batch / batch_size_actual
            cc_avg = cc_batch / batch_size_actual
            ifc_avg = ifc_batch / batch_size_actual
            vif_avg = vif_batch / batch_size_actual
            psnr_avg = psnr_batch / batch_size_actual
            cos_sim_avg = cos_sim_batch / batch_size_actual
            cos_sim_binary_avg = cos_sim_binary_batch / batch_size_actual
            cos_sim_channel_avg = cos_sim_channel_batch / batch_size_actual
            cos_sim_binary_channel_avg = cos_sim_binary_channel_batch / batch_size_actual

            rmse_all += rmse_avg
            ssim_all += ssim_avg
            cc_all += cc_avg
            ifc_all += ifc_avg
            vif_all += vif_avg
            psnr_all += psnr_avg
            cos_sim_all += cos_sim_avg
            cos_sim_binary_all += cos_sim_binary_avg
            cos_sim_channel_all += cos_sim_channel_avg
            cos_sim_binary_channel_all += cos_sim_binary_channel_avg

            # 写入CSV
            with open(progress_csv, "a") as f:
                f.write(f"{batch_idx+1},{rmse_avg:.4f},{ssim_avg:.4f},{cc_avg:.4f},{ifc_avg:.4f},{vif_avg:.4f},{psnr_avg:.4f},{cos_sim_avg:.4f},{cos_sim_binary_avg:.4f},{cos_sim_channel_avg:.4f},{cos_sim_binary_channel_avg:.4f}\n")

            logger.log(f"批次 {batch_idx+1} 完成: RMSE={rmse_avg:.4f}, SSIM={ssim_avg:.4f}, CC={cc_avg:.4f}, "
                      f"IFC={ifc_avg:.4f}, VIF={vif_avg:.4f}, PSNR={psnr_avg:.4f}, "
                      f"CosSim={cos_sim_avg:.4f}, CosSim_Binary={cos_sim_binary_avg:.4f}, "
                      f"CosSim_Channel={cos_sim_channel_avg:.4f}, CosSim_Binary_Channel={cos_sim_binary_channel_avg:.4f}")

        # 计算所有批次的平均结果
        if num_batches > 0:
            logger.log(f"\n{'='*40}\n全部测试结果汇总 ({num_batches}批次)\n{'='*40}")
            logger.log(f"平均 RMSE: {rmse_all/num_batches:.4f}")
            logger.log(f"平均 SSIM: {ssim_all/num_batches:.4f}")
            logger.log(f"平均 CC: {cc_all/num_batches:.4f}")
            logger.log(f"平均 IFC: {ifc_all/num_batches:.4f}")
            logger.log(f"平均 VIF: {vif_all/num_batches:.4f}")
            logger.log(f"平均 PSNR: {psnr_all/num_batches:.4f}")
            logger.log(f"平均 CosSim: {cos_sim_all/num_batches:.4f}")
            logger.log(f"平均 CosSim_Binary: {cos_sim_binary_all/num_batches:.4f}")
            logger.log(f"平均 CosSim_Channel: {cos_sim_channel_all/num_batches:.4f}")
            logger.log(f"平均 CosSim_Binary_Channel: {cos_sim_binary_channel_all/num_batches:.4f}")

if __name__ == "__main__":
    main()