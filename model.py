import copy
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np

# def run_network(pcd, pth_file):
#     # 示例：复制输入点云
#     print(f"Running network with {pth_file}...")
#     generated_pcd = copy.deepcopy(pcd)
#     # 用户应在此处调用实际的网络
#
#
#
#     return generated_pcd


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()
        # num_dense、latent_dim、grid_size是网络的基本参数，分别代表稠密点云的点数、潜在维度和网格大小
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        # 先算**，后算%。判断self.num_dense（稠密点云的点数）能够被self.grid_size ** 2（网格大小的平方）整除
        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)  # 计算出粗糙点云的数

        # 特征提取，两个PN层：first_conv、second_conv
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        # 最终的细化步骤
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape  # 输入xyz是批量的点云数据，形状为(B, N, _)，其中B是批量大小，N是点数

        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))  # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)

        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)  # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class AdaPoinTr(nn.Module):
    pass


def run_network(pcd, pth_file):
    # 示例：复制输入点云
    global model
    print(f"Running network with {pth_file}...")
    if pth_file.endswith("PCN.pth"):
        print("模型是PCN.pth")
        model = PCN()
    if pth_file.endswith("AdaPoinTr.pth"):
        print("模型是AdaPoinTr.pth")
        model = AdaPoinTr()
   # model = PCN()
    print("111111")
    model.load_state_dict(torch.load(pth_file))
    print("222222")
    device = torch.device('cuda')
    print("33333")
    model.to(device)
    print("44444")
    model.eval()
    print("55555")
    # with torch.no_grad():
    #     print("66666")
    #     point_cloud = pcd.to(device)
    #     print("7777777")
    #     _, pcd = model(point_cloud)
    #     print("88888")
    #     pcd = pcd.detach().cpu().numpy()
    #     print("99999")
    # #  generated_pcd = copy.deepcopy(pcd)
    # # 用户应在此处调用实际的网络
    #
    # return pcd

    # Convert open3d point cloud to numpy array
    pcd_np = np.asarray(pcd.points, dtype=np.float32)
    print("66")
    # Ensure the point cloud has the correct shape (B, N, 3)
    if pcd_np.ndim == 2:
        pcd_np = np.expand_dims(pcd_np, axis=0)
    print("77")
    # Convert numpy array to PyTorch tensor and move to device
    point_cloud = torch.from_numpy(pcd_np).to(device)
    print("88")
    with torch.no_grad():
        print("99")
        if pth_file.endswith("PCN.pth"):
            _, generated_pcd = model(point_cloud)
        if pth_file.endswith("AdaPoinTr.pth"):
            generated_pcd = model(point_cloud)[1]
        print("10  10")
        generated_pcd = generated_pcd.detach().cpu().numpy()
        print("11  11")
    # Ensure the output has the correct shape (N, 3)
    print("12  12")
    if generated_pcd.ndim == 3:
        print("13  13")
        generated_pcd = generated_pcd.squeeze(0)
        print("14  14")
    # Convert numpy array back to open3d point cloud
    print("15  15")
    generated_pcd_o3d = o3d.geometry.PointCloud()
    print("16 16")
    generated_pcd_o3d.points = o3d.utility.Vector3dVector(generated_pcd)
    print("17 17")
    return generated_pcd_o3d



if __name__ == '__main__':
    run_network('E:/Code/keshihua/PCN_data.ply', 'E:/Code/keshihua/off_PCN.pth')