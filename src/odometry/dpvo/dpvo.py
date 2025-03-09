import torch
import numpy as np
import os
import torch.nn.functional as F
import subprocess as sp

from odometry.dpvo import fastba
from odometry.dpvo import altcorr
from odometry.dpvo import lietorch
from odometry.dpvo.lietorch import SE3

from odometry.dpvo.net import VONet
from odometry.dpvo.utils import *
from odometry.dpvo import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device=torch.device("cuda:0"))

class DPVO:
    def __init__(self, cfg, network, device, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.device = device
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES
        Id = SE3.Identity(1, device=self.device)
        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device=self.device)
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device=self.device)
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device=self.device)

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device=self.device)
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device=self.device)

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device=self.device)
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device=self.device)

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.kk = torch.as_tensor([], dtype=torch.long, device=self.device)
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.update_step = 0
        self.viewer = None
        if viz:
            self.start_viewer()

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.to(self.device)
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device=self.device)

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            # print(self.traj[t])
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def dump_poses(self, dir):
        """ interpolate missing poses """
        #NOTE: poses will be dumped for all the frames if args.stride==1
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        np.save(os.path.join('results', 'dpvo', 'poses', dir), poses, allow_pickle=True)

    def get_poses(self):
        """ interpolate missing poses """
        #NOTE: poses will be dumped for all the frames if args.stride==1
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float32) 
        
        return poses, tstamps
    
    def get_patch_depths(self):
        #NOTE: only dumping patch depths for the first frame of the sequence
        start_frame_patches = self.patches_[0]
        x, y, inv_depths = start_frame_patches[:, :, 1, 1].T
        x = x.data.cpu().numpy()
        y = y.data.cpu().numpy()
        inv_depths = inv_depths.data.cpu().numpy()      
        return x, y, inv_depths

    def dump_patch_depths(self, dir):
        #NOTE: only dumping patch depths for the first frame of the sequence
        start_frame_patches = self.patches_[0]
        x, y, inv_depths = start_frame_patches[:, :, 1, 1].T
        x = x.data.cpu().numpy()
        y = y.data.cpu().numpy()
        inv_depths = inv_depths.data.cpu().numpy()      
        np.save(os.path.join('results', 'dpvo', 'depths', dir), (x, y, inv_depths), allow_pickle=True)

        return x, y, inv_depths

    def terminate(self):
        if self.viewer is not None:
            # self.viewer.close()
            print(self.viewer.join())

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device=self.device)
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device=self.device)
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            print(f"[{self.update_step}] Updating pcd - currently there are {len(points)} points")
            
            self.points_[:len(points)] = points[:]
            # To check how many patches are non-zero:
            # ((self.patches[0] != 0).view(self.patches[0].size(0), -1).sum(dim=1) > 0).sum()
            # print(self.points_.shape)
            # print(self.patches.shape)
            self.update_step += 1
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device=self.device),
            torch.arange(0, self.n, device=self.device), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(self.n-1, self.n, device=self.device), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device=self.device),
            torch.arange(max(self.n-r, 0), self.n, device=self.device), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """
        updateFlag = False

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)
            self.viewer.loop()

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        # print(f"DEBUG: Image:", image)

        # FOR DEBUGGING: Something seems to be wrong here, because patches are always set to all zeros
        # while debugging.
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    gradient_bias=self.cfg.GRADIENT_BIAS, 
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter

        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        # print(self.imap_, self.gmap_, self.fmap1_, self.fmap2_)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            motion_magnitude = self.motion_probe()
            # print(f"DEBUG: Predicted motion magnitude: {motion_magnitude}")
            if motion_magnitude < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        # print(f"DEBUG: self.n = {self.n}")
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # First update is done after 8 frames, to be sure that there is enough motion for optimization
        if self.n == 8 and not self.is_initialized:
            # print("[INFO] Reaching step 8, now run 12 optimization steps.")
            self.is_initialized = True            

            # After colleting the first 8 frames, the update method for optimization is called 12 times
            for itr in range(12):
                self.update()
            
            updateFlag = True

        # When self.n > 8, besides updating the poses and depths with an optimization step, 
        # the keyframe method is also called to possibly delete some frames from the keyframe list.
        elif self.is_initialized:
            # print("[INFO] Step > 8, update for each incoming frame, plus keyframe checking.")
            self.update()
            updateFlag = True
            self.keyframe()
        else:
            # print("[INFO] Adding edges to connectivity graph, no update.")
            pass
        
        return updateFlag, self.get_poses()