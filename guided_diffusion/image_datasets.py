import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps
import scipy.sparse as sp_sparse
def np_norm(inp):
    max_in=np.max(inp)
    min_in = np.min(inp)
    return (inp-min_in)/(max_in-min_in)

def gray_value_of_gene(gene_class,gene_order):
    gene_order=list(gene_order)
    Index=gene_order.index(gene_class)
    interval=255/len(gene_order)
    value=Index*interval
    return int(value)

import numpy as np


# def gray_value_of_gene(gene_class, gene_order):
#     gene_order = list(gene_order)
#     if isinstance(gene_class, (list, tuple, set)):
#         result_values = []
#         for single_gene_class in gene_class:
#             try:
#                 Index = gene_order.index(single_gene_class)
#                 interval = 255 / len(gene_order)
#                 value = Index * interval
#                 result_values.append(int(value))
#             except ValueError:
#                 print(f"{single_gene_class} not found in gene_order, skipping.")
#         return result_values
#     else:
#         try:
#             Index = gene_order.index(gene_class)
#             interval = 255 / len(gene_order)
#             value = Index * interval
#             return [int(value)]
#         except ValueError:
#             print(f"{gene_class} not found in gene_order, returning default value.")
#             return [0]  # 返回默认值列表形式，方便后续统一处理


# def load_data(data_root,dataset_use,status,SR_times,gene_num
# ):

#     if dataset_use=='Xenium':
#         dataset= Xenium_dataset(data_root,SR_times,status,gene_num)
#     elif dataset_use=='Visium':
#         dataset= Visium_dataset(data_root,gene_num)
#     elif dataset_use=='NBME':
#         dataset= NBME_dataset(data_root,gene_num)

#     return dataset

# def load_data(data_root,dataset_use,status,SR_times,gene_num
# ):

#     if dataset_use=='Xenium':
#         dataset= Xenium_dataset(data_root,SR_times,status,gene_num)
#     elif dataset_use=='SGE':
#         dataset= SGE_dataset(data_root,gene_num)
#     elif dataset_use=='BreastST':
#         dataset= BreastST_dataset(data_root,gene_num)

#     return dataset

# class Xenium_dataset(Dataset):
#     def __init__(self, data_root,SR_times,status,gene_num):
#         '''
#             data_root: 数据根目录的路径。
#             SR_times: 下采样倍数，影响加载的 HR ST 数据的分辨率。
#             status: 指定数据集的状态，值为 'Train' 或 'Test'，用于选择不同的样本。
#             gene_num: 需要处理的基因数量。
#         '''
#         if status=='Train':
#             sample_name=['01220101', '01220102', 'NC1', 'NC2', '0418']#, '0418'
#             # sample_name = ['NC2']
#         elif status=='Test':
#             sample_name = ['01220201', '01220202']
#             #sample_name = ['01220202']
       
#         SR_ST_all=[]
#         ### HR ST
#         # for sample_id in sample_name:
#         #     sub_patches=os.listdir(data_root+'Xenium/HR_ST/extract/'+sample_id)
#         #     for patch_id in sub_patches:
#         #         if SR_times==10:
#         #             SR_ST=np.load(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npy')
#         #         elif SR_times==5:
#         #             SR_ST = np.load(data_root + 'Xenium/HR_ST/extract/' + sample_id + '/' + patch_id + '/HR_ST_128.npy')
#         #         #print(type(SR_ST))
#         #         SR_ST=np.transpose(SR_ST,axes=(2,0,1))
#         #         SR_ST_all.append(SR_ST)
#         for sample_id in sample_name:
#             sub_patches=os.listdir(data_root+'Xenium/HR_ST1/extract/'+sample_id)
#             for patch_id in sub_patches:
#                 if SR_times==10:
#                     SR_ST=sp_sparse.load_npz(data_root+'Xenium/HR_ST1/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz').toarray().reshape(256, 256, 280)
#                 elif SR_times==5:
#                     SR_ST = sp_sparse.load_npz(data_root+'Xenium/HR_ST1/extract/'+sample_id+'/'+patch_id+'/HR_ST_128.npz').toarray().reshape(128, 128, 280)
#                 SR_ST=np.transpose(SR_ST,axes=(2,0,1))
#                 SR_ST_all.append(SR_ST)
#         SR_ST_all=np.array(SR_ST_all)
#         SR_ST_all=np.array(SR_ST_all)
#         Sum=np.sum(SR_ST_all,axis=(0,2,3))
#         # gene_order=np.argsort(Sum)[::-1][0:gene_num]
        
#         gene_order=np.load(data_root + 'gene_order.npy')[0:gene_num]
#         self.SR_ST_all=SR_ST_all[:,gene_order,...].astype(np.float64) # (X,50,256,256)

#         # Sum_gene = np.sum(SR_ST_all, axis=(0, 2, 3))
#         # gene_coexpre=np.zeros(shape=(gene_num,gene_num))
#         # for i in range(gene_num):
#         #     for j in range(gene_num):
#         #         gene_coexpre[i,j]=Sum_gene[i]/Sum_gene[j]
#         # np.save(data_root + 'gene_coexpre.npy',gene_coexpre)

#         ####### norm
#         # Z=np.sum(self.SR_ST_all)
#         # self.SR_ST_all = np.log((1 + self.SR_ST_all) / (Z))

#         # for ii in range(self.SR_ST_all.shape[0]):
#         #     Max=np.max(self.SR_ST_all[ii])
#         #     Min=np.min(self.SR_ST_all[ii])
#         #     self.SR_ST_all[ii]=(self.SR_ST_all[ii]-Min)/(Max-Min)

#         for ii in range(self.SR_ST_all.shape[0]): #初始化 SR_ST_all 列表，用于存储加载的 HR ST 数据。
#             for jj in range(self.SR_ST_all.shape[1]):
#                 if np.sum(self.SR_ST_all[ii, jj]) != 0:
#                     Max=np.max(self.SR_ST_all[ii,jj])
#                     Min=np.min(self.SR_ST_all[ii,jj])
                    
#                     self.SR_ST_all[ii,jj]=(self.SR_ST_all[ii,jj]-Min)/(Max-Min)#对选择的基因进行归一化处理
                    
#         ### spot ST
#         spot_ST_all=[]
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
#                 spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
#                 spot_ST_all.append(spot_ST)
#         spot_ST_all = np.array(spot_ST_all)
#         self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)

#         ####### norm
#         # Z = np.sum(self.spot_ST_all)
#         # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

#         # for ii in range(self.spot_ST_all.shape[0]):
#         #     Max=np.max(self.spot_ST_all[ii])
#         #     Min=np.min(self.spot_ST_all[ii])
#         #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

#         for ii in range(self.spot_ST_all.shape[0]):
#             for jj in range(self.spot_ST_all.shape[1]):
#                 if np.sum(self.spot_ST_all[ii, jj]) != 0:
#                     Max = np.max(self.spot_ST_all[ii,jj])
#                     Min = np.min(self.spot_ST_all[ii,jj])
#                     self.spot_ST_all[ii,jj] = (self.spot_ST_all[ii,jj] - Min) / (Max - Min)

#         ### WSI 5120
#         WSI_5120_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
#                 max=np.max(WSI_5120)
#                 WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
#                 WSI_5120_all.append(WSI_5120)
#         self.WSI_5120_all = np.array(WSI_5120_all)

#         ### WSI 320
#         self.num_320=[]
#         WSI_320_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_320 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
#                 #WSI_320 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/320.npy')

#                 # WSI_320_to16=[]
#                 # for i in range(WSI_320.shape[0]):
#                 #     im = Image.fromarray(WSI_320[i])
#                 #     im = im.resize((16, 16))
#                 #     im = np.asarray(im)
#                 #     WSI_320_to16.append(im)
#                 #
#                 # WSI_320_to16_NEW=WSI_320_to16
#                 # times = int(np.floor(256 / WSI_320.shape[0]))
#                 # remaining = 256 % WSI_320.shape[0]
#                 # if times > 1:
#                 #     for k in range(times - 1):
#                 #         WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16
#                 # if not remaining == 0:
#                 #     WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16[0:remaining]
#                 # WSI_320_to16_NEW = np.array(WSI_320_to16_NEW)
#                 # entropy=[]
#                 # for i in range(WSI_320_to16_NEW.shape[0]):
#                 #     entropy.append(np.round(shannon_entropy(WSI_320_to16_NEW[i])))
#                 # entropy=np.array(entropy)
#                 # entropy_order=np.argsort(entropy)
#                 # WSI_320_to16_entropy=WSI_320_to16_NEW[entropy_order]
#                 # np.save(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy',WSI_320_to16_entropy)
#                 # print(patch_id)

#                 WSI_320 = np.transpose(WSI_320, axes=(0, 3,1,2))
#                 WSI_320_all.append(WSI_320)
#         self.WSI_320_all = np.array(WSI_320_all)
#         max_320=np.max(WSI_320)
#         a=1

#         print(self.SR_ST_all.shape)
#         print(self.spot_ST_all.shape)
#         print(self.WSI_5120_all.shape)
#         print(self.WSI_320_all.shape)

#     def __len__(self):
#         return self.WSI_320_all.shape[0]

#     def __getitem__(self, index):
#         '''
#             self.SR_ST_all[index]: 对应的 HR ST 数据。
#             self.spot_ST_all[index]: 对应的点位 ST 数据。
#             self.WSI_5120_all[index]: 对应的 WSI 5120 数据。
#             self.WSI_320_all[index]: 对应的 WSI 320 数据。
#         '''

#         return self.SR_ST_all[index], self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


# class Visium_dataset(Dataset):
#     def __init__(self, data_root,gene_num):

#         sample_name = ['0701', '0106']

#         gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
#         ### spot ST
#         spot_ST_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Visium/spot_ST/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 spot_ST = np.load(data_root + 'Visium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
#                 spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
#                 spot_ST_all.append(spot_ST)
#         spot_ST_all = np.array(spot_ST_all)
#         self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)



#         ####### norm
#         # Z = np.sum(self.spot_ST_all)
#         # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

#         # for ii in range(self.spot_ST_all.shape[0]):
#         #     Max=np.max(self.spot_ST_all[ii])
#         #     Min=np.min(self.spot_ST_all[ii])
#         #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

#         for ii in range(self.spot_ST_all.shape[0]):
#             for jj in range(self.spot_ST_all.shape[1]):
#                 if np.sum(self.spot_ST_all[ii,jj])!=0:
#                     Max = np.max(self.spot_ST_all[ii, jj])
#                     Min = np.min(self.spot_ST_all[ii, jj])
#                     self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)



#         ### WSI 5120
#         WSI_5120_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_5120 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
#                 max = np.max(WSI_5120)
#                 WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
#                 WSI_5120_all.append(WSI_5120)
#         self.WSI_5120_all = np.array(WSI_5120_all)


#         ### WSI 320
#         self.num_320 = []
#         WSI_320_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_320 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
#                 WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
#                 WSI_320_all.append(WSI_320)
#         self.WSI_320_all = np.array(WSI_320_all)
#         max_320 = np.max(WSI_320)
#         a = 1

#         a=1


#     def __len__(self):
#         return self.WSI_320_all.shape[0]

#     def __getitem__(self, index):
#         return  self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


# class NBME_dataset(Dataset):
#     def __init__(self, data_root,gene_num):
#         sample_name = os.listdir(data_root + 'NBME/spot_ST/extract/')

#         gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
#         ### spot ST
#         spot_ST_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'NBME/spot_ST/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 spot_ST = np.load(data_root + 'NBME/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
#                 spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
#                 spot_ST_all.append(spot_ST)
#         spot_ST_all = np.array(spot_ST_all)
#         self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)



#         ####### norm
#         # Z = np.sum(self.spot_ST_all)
#         # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

#         # for ii in range(self.spot_ST_all.shape[0]):
#         #     Max=np.max(self.spot_ST_all[ii])
#         #     Min=np.min(self.spot_ST_all[ii])
#         #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

#         for ii in range(self.spot_ST_all.shape[0]):
#             for jj in range(self.spot_ST_all.shape[1]):
#                 if np.sum(self.spot_ST_all[ii, jj]) != 0:
#                     Max = np.max(self.spot_ST_all[ii, jj])
#                     Min = np.min(self.spot_ST_all[ii, jj])
#                     self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

#         ### WSI 5120
#         WSI_5120_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_5120 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
#                 max = np.max(WSI_5120)
#                 WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
#                 WSI_5120_all.append(WSI_5120)
#         self.WSI_5120_all = np.array(WSI_5120_all)



#         ### WSI 320
#         self.num_320 = []
#         WSI_320_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_320 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
#                 WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))

#                 WSI_320_all.append(WSI_320)
#         self.WSI_320_all = np.array(WSI_320_all)
#         max_320 = np.max(WSI_320)



#         a = 1
        
#     def __len__(self):
#         return self.WSI_320_all.shape[0]

#     def __getitem__(self, index):
#         return self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]

import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps

def load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene
):

    if dataset_use=='Xenium':
        dataset= Xenium_dataset(data_root,SR_times,status,gene_num,all_gene)
    elif dataset_use=='SGE':
        dataset= SGE_dataset(data_root,gene_num,all_gene)
    elif dataset_use=='BreastST':
        dataset= BreastST_dataset(data_root,gene_num,all_gene)

    return dataset

class Xenium_dataset(Dataset):
    def __init__(self, data_root, SR_times, status, gene_num,all_gene):
        '''
            data_root: 数据根目录的路径。
            SR_times: 下采样倍数，影响加载的HR ST数据的分辨率。
            status: 指定数据集的状态，值为 'Train' 或 'Test'，用于选择不同的样本。
            gene_num: 需要处理的基因数量。
        '''
        if status == 'Train':
            sample_name = ['01220101', '01220102', 'NC1', 'NC2', '0418']

        elif status == 'Test':
            sample_name = ['01220201', '01220202']

        self.all_gene = all_gene
        gene_order_path = os.path.join(data_root, 'gene_order.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        self.gene_order = gene_order
        SR_ST_all = []
        self.gene_scale = []#new1.5:
        self.gene_num = gene_num#new1.5:
        
        # 
        # 
        ### HR ST
        for sample_id in sample_name:
            sub_patches=os.listdir(data_root+'Xenium/HR_ST1/extract/'+sample_id)
            for patch_id in sub_patches:
                if SR_times==10:
                    SR_ST=sp_sparse.load_npz(data_root+'Xenium/HR_ST1/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz').toarray().reshape(256, 256, 280)
                elif SR_times==5:
                    SR_ST = sp_sparse.load_npz(data_root+'Xenium/HR_ST1/extract/'+sample_id+'/'+patch_id+'/HR_ST_128.npz').toarray().reshape(128, 128, 280)
                SR_ST=np.transpose(SR_ST,axes=(2,0,1))
                SR_ST_all.append(SR_ST)
                self.gene_scale.append(gene_order)#new1.5:
        self.SR_ST_all = np.array(SR_ST_all)
        #new1.5:
        self.gene_scale = np.array(self.gene_scale)
        self.gene_scale_groups = np.reshape(self.gene_scale, (self.gene_scale.shape[0]*(all_gene//gene_num),gene_num))
        #new1.5:
        # 新
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)

        # print('SR_ST_all',self.SR_ST_all.shape)
        self.SR_ST_all_groups = np.reshape(self.SR_ST_all, (self.SR_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.SR_ST_all.shape[2],self.SR_ST_all.shape[3]))
        # print('SR_ST_all_groups',self.SR_ST_all_groups.shape)
        for ii in range(self.SR_ST_all_groups.shape[0]):  # 初始化SR_ST_all列表，用于存储加载的HR ST数据。
            for jj in range(self.SR_ST_all_groups.shape[1]):
                if np.sum(self.SR_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.SR_ST_all_groups[ii, jj])
                    Min = np.min(self.SR_ST_all_groups[ii, jj])
                    self.SR_ST_all_groups[ii, jj] = (self.SR_ST_all_groups[ii, jj] - Min) / (Max - Min)  # 对选择的基因进行归一化处理

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        # print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all_groups = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        # print('spot_ST_all_groups',self.spot_ST_all_groups.shape)
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        self.WSI_5120_all_expanded = np.repeat(self.WSI_5120_all, all_gene//gene_num, axis = 0)
        # self.WSI_5120_all_expanded = []
        # for _ in range(all_gene//gene_num):
        #     self.WSI_5120_all_expanded.append(self.WSI_5120_all)
        # self.WSI_5120_all_expanded = np.concatenate(self.WSI_5120_all_expanded, axis=0)

        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        self.WSI_320_all_expanded = np.repeat(self.WSI_320_all, all_gene//gene_num, axis = 0)
        # 对WSI 320数据按照分组数量复制到batch维度
        # self.WSI_320_all_expanded = []
        # for _ in range(all_gene//gene_num):
        #     self.WSI_320_all_expanded.append(self.WSI_320_all)
        # self.WSI_320_all_expanded = np.concatenate(self.WSI_320_all_expanded, axis=0)
        # max_320 = np.max(WSI_320)
        a = 1

        # print(self.SR_ST_all_groups.shape)
        # print(self.spot_ST_all_groups.shape)
        # print(self.WSI_5120_all_expanded.shape)
        # print(self.WSI_320_all_expanded.shape)
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.SR_ST_all_groups.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        ''' 
        #new1.5:
        gene_class = self.gene_scale_groups[index]

        Gene_index_maps = []
        # print('gene_class',gene_class)
        # print('self.gene_order',self.gene_order)
        for gene_code in gene_class:
            Gene_codes = gray_value_of_gene(gene_code, self.gene_order)
            # print(Gene_codes,end=' ')
            Gene_index_map = np.ones(shape=(256, 256,1)) * Gene_codes / 255.0
            Gene_index_maps.append(Gene_index_map)
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2)
        final_Gene_index_map = np.moveaxis(final_Gene_index_map, 2, 0)
        #new1.5:

        return self.SR_ST_all_groups[index], self.spot_ST_all_groups[index], self.WSI_5120_all_expanded[index], self.WSI_320_all_expanded[index], gene_class, final_Gene_index_map #new1.5:



class SGE_dataset(Dataset):
    def __init__(self, data_root, gene_num):
        sample_name = ['0701', '0106']

        # 加载全部基因顺序信息，不再截取前gene_num个
        all_gene=30
        gene_order_path = os.path.join(data_root, 'gene_order.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        SR_ST_all = []

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Visium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all_groups = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        print('spot_ST_all_groups',self.spot_ST_all_groups.shape)
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        self.WSI_5120_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all_expanded.append(self.WSI_5120_all)
        self.WSI_5120_all_expanded = np.concatenate(self.WSI_5120_all_expanded, axis=0)

        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)

        # 对WSI 320数据按照分组数量复制到batch维度
        self.WSI_320_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all_expanded.append(self.WSI_320_all)
        self.WSI_320_all_expanded = np.concatenate(self.WSI_320_all_expanded, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all_groups.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all_groups[index], self.WSI_5120_all_expanded[index], self.WSI_320_all_expanded[index]


class BreastST_dataset(Dataset):
    def __init__(self, data_root, gene_num):
        from skimage.transform import resize
        sample_name = os.listdir(data_root + 'NBME/spot_ST/extract/')

        all_gene=130
        gene_order_path = os.path.join(data_root, 'gene_order.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        SR_ST_all = []

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'NBME/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all_groups = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        print('spot_ST_all_groups',self.spot_ST_all_groups.shape)
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        self.WSI_5120_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all_expanded.append(self.WSI_5120_all)
        self.WSI_5120_all_expanded = np.concatenate(self.WSI_5120_all_expanded, axis=0)

        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)

        # 对WSI 320数据按照分组数量复制到batch维度
        self.WSI_320_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all_expanded.append(self.WSI_320_all)
        self.WSI_320_all_expanded = np.concatenate(self.WSI_320_all_expanded, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all_groups.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all_groups[index], self.WSI_5120_all_expanded[index], self.WSI_320_all_expanded[index]


if  __name__ == '__main__':
    import cv2
    data_root = '/home/zeiler/ST_proj/data/Breast_cancer/'
    dataset_use = 'Xenium'#BreastST,SGE
    status = 'Test'
    SR_times = 10
    gene_num = 10
    all_gene=30
    dataset = load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    for idx,(SR_ST_all,LR_ST_all, WSI_5120_all,WSI_320_all, gene_captions, Gene_index_map) in enumerate(dataloader):
        SR_ST_all = SR_ST_all.numpy()
        SR_ST = SR_ST_all[0]
        LR_ST_all = LR_ST_all.numpy()
        LR_ST = LR_ST_all[0]
        WSI_5120_all = WSI_5120_all.numpy()
        WSI = WSI_5120_all[0]
        WSI = np.transpose(WSI, axes=(1, 2, 0))
        # WSI_5120_all = WSI_5120_all.numpy()
        wsi_path = f'temp4/{idx+1}_gene_WSI.png'
        plt.imsave(wsi_path, WSI)
        for k in range(SR_ST.shape[0]):
            gt_path = f'temp4/{idx+1}_gene_{k+1}_GT.png'
            lr_path = f'temp4/{idx+1}_gene_{k+1}_LR.png'
            # pred_path = f'temp/patch_{idx+1}_gene_{k+1}.png'
            plt.imsave(gt_path, SR_ST[k], cmap='viridis')
            # 对LR_ST[k]进行分辨率放大
            LR_ST_resized = cv2.resize(LR_ST[k], (256, 256), interpolation=cv2.INTER_LINEAR)
            plt.imsave(lr_path, LR_ST_resized, cmap='viridis')
            # plt.imsave(pred_path, pred_sample[k], cmap='viridis')
            


