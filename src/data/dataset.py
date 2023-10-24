import numpy as np
from typing import List, Type
from torch.utils.data import Dataset, DataLoader, sampler
import os
import glob
import torch
import utils.quat_util as Q
import utils.file_util as F
from analytical.optimal_svd import direct_SVD
from scipy.spatial.transform import Rotation as R

class ModelNetDataset(Dataset):
    """
    Dataset to load ModelNet40 mesh data
    """
    def __init__(self, base_path: str, category_list: list, num_sample: int, 
                sigma: float, num_rot: int, range_max: int, range_min: int,
                rot_option:str, trans_max:float=0.5, angle_max:int=180):
        all_files = []
        for c in category_list:
            curr_path = "/".join([base_path, c, "train"])
            curr_list = F.list_files_in_dir(curr_path)
            all_files += curr_list

        self.select_files = []
        for f in all_files:
            curr_vert = F.read_off_file(f)
            if len(curr_vert) > range_max or len(curr_vert) < range_min:
                continue
            self.select_files.append(f)

        self.sigma = sigma
        self.num_sample = num_sample
        self.num_rot = num_rot
        self.t_max = trans_max
        self.r_max = angle_max
        self.rot_option = rot_option
        
    def __len__(self):
        return self.num_rot

    def __getitem__(self, index: int):
        #select one cloud from all selected files
        random_pick = np.random.randint(len(self.select_files))
        orig_cloud = torch.as_tensor(F.read_off_file(self.select_files[random_pick]), dtype=torch.float32)

        #downsampling with random point indices
        random_indices = torch.randperm(len(orig_cloud))
        num_points = int(self.num_sample)
        picked_indices = random_indices[:num_points]  
        source_cloud = orig_cloud[picked_indices]

        if self.rot_option == "JPL":
            curr_rot = Q.generate_random_quat()
            r = R.from_quat(curr_rot)
        elif self.rot_option == "Hamitonian":
            angle = Q.generate_random_rot(self.r_max)
            r = R.from_euler('zyx', angle, degrees=True)

        rot_mat = r.as_matrix()
        rot_mat_tensor = torch.as_tensor(rot_mat, dtype=torch.float32)

        rotate_cloud = torch.matmul(source_cloud, rot_mat_tensor)
        noise = self.sigma*torch.randn_like(source_cloud)
        target_cloud = rotate_cloud + noise

        translation_ab = torch.rand(3, dtype=torch.float32) - self.t_max
        shift_cloud = target_cloud + translation_ab

        
        concatenate_cloud = torch.empty(2, num_points, 3, dtype=torch.float32)
        
        concatenate_cloud[0,:,:] = source_cloud
        concatenate_cloud[1,:,:] = shift_cloud

        # we use convention of DCP, that the input is in shape (b,3,n_points)
        concatenate_cloud = concatenate_cloud.transpose(2,1)

        return concatenate_cloud, torch.as_tensor(r.as_quat(),dtype=torch.float32), translation_ab

class KittiOdometryDataset(Dataset):
    """
    Dataset to load kitti' velodyne lidar data of odometry
    """
    def __init__(self, base_path:str, seq_num:int) -> None:
        super().__init__()
        self.velo_path = F.get_all_bins(base_path + seq_num)

    def __len__(self):
        return len(self.velo_path)

    def __getitem__(self, index: int):
        return F.get_velo(self.velo_path[index])