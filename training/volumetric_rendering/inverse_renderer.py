import torch
import torch.nn.functional as F
import neural_renderer as nr
import math

EPS = 1e-7

def mm_normalize(x, min=0, max=1):
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    x_z = (x - x_min) / x_range
    x_out = x_z * (max - min) + min
    return x_out


def rand_range(size, min, max):
    return torch.rand(size)*(max-min)+min


def rand_posneg_range(size, min, max):
    i = (torch.rand(size) > 0.5).type(torch.float)*2.-1.
    return i*rand_range(size, min, max)


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def get_transform_matrices(view):
    b = view.size(0)
    if view.size(1) == 6:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = view[:,3:].reshape(b,1,3)
    elif view.size(1) == 5:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        delta_xy = view[:,3:].reshape(b,1,2)
        trans_xyz = torch.cat([delta_xy, torch.zeros(b,1,1).to(view.device)], 2)
    elif view.size(1) == 3:
        rx = view[:,0]
        ry = view[:,1]
        rz = view[:,2]
        trans_xyz = torch.zeros(b,1,3).to(view.device)
    rot_mat = get_rotation_matrix(rx, ry, rz)
    return rot_mat, trans_xyz


def get_face_idx(b, h, w):
    idx_map = torch.arange(h*w).reshape(h,w)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map[:h-1,1:]], -1).reshape(-1,3)
    faces2 = torch.stack([idx_map[:h-1,1:], idx_map[1:,:w-1], idx_map[1:,1:]], -1).reshape(-1,3)
    return torch.cat([faces1,faces2], 0).repeat(b,1,1).int()


def vcolor_to_texture_cube(vcolors):
    # input bxcxnx3
    b, c, n, f = vcolors.shape
    coeffs = torch.FloatTensor(
        [[ 0.5,  0.5,  0.5],
         [ 0. ,  0. ,  1. ],
         [ 0. ,  1. ,  0. ],
         [-0.5,  0.5,  0.5],
         [ 1. ,  0. ,  0. ],
         [ 0.5, -0.5,  0.5],
         [ 0.5,  0.5, -0.5],
         [ 0. ,  0. ,  0. ]]).to(vcolors.device)
    return coeffs.matmul(vcolors.permute(0,2,3,1)).reshape(b,n,2,2,2,c)


def get_textures_from_im(im, tx_size=1):
    b, c, h, w = im.shape
    if tx_size == 1:
        textures = torch.cat([im[:,:,:h-1,:w-1].reshape(b,c,-1), im[:,:,1:,1:].reshape(b,c,-1)], 2)
        textures = textures.transpose(2,1).reshape(b,-1,1,1,1,c)
    elif tx_size == 2:
        textures1 = torch.stack([im[:,:,:h-1,:w-1], im[:,:,:h-1,1:], im[:,:,1:,:w-1]], -1).reshape(b,c,-1,3)
        textures2 = torch.stack([im[:,:,1:,:w-1], im[:,:,:h-1,1:], im[:,:,1:,1:]], -1).reshape(b,c,-1,3)
        textures = vcolor_to_texture_cube(torch.cat([textures1, textures2], 2)) # bxnx2x2x2xc
    else:
        raise NotImplementedError("Currently support texture size of 1 or 2 only.")
    return textures


class Renderer():
    def __init__(self,
                 resolution,
                 min_depth,
                 max_depth,
                 rot_center_depth,
                 fov,
                 tex_cube_size,
                 renderer_min_depth,
                 renderer_max_depth, 
                 device=None):
        self.resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.rot_center_depth = rot_center_depth
        self.fov = fov
        self.tex_cube_size = tex_cube_size
        self.renderer_min_depth = renderer_min_depth
        self.renderer_max_depth = renderer_max_depth

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        ## renderer for visualization
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = torch.FloatTensor(R).to(device)
        t = torch.zeros(1,3, dtype=torch.float32).to(device)
        fx = (self.resolution-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.resolution-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.resolution-1)/2
        cy = (self.resolution-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(device)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)
        self.renderer = nr.Renderer(camera_mode='projection',
                                    light_intensity_ambient=1.0,
                                    light_intensity_directional=0.,
                                    K=self.K, R=R, t=t,
                                    near=self.renderer_min_depth, far=self.renderer_max_depth,
                                    image_size=self.resolution, orig_size=self.resolution,
                                    fill_back=True,
                                    background_color=[1,1,1])

    def set_transform_matrices(self, view):
        self.rot_mat, self.trans_xyz = get_transform_matrices(view)

    def rotate_pts(self, pts, rot_mat):
        centroid = torch.FloatTensor([0.,0.,self.rot_center_depth]).to(pts.device).view(1,1,3)
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose(2,1))  # rotate
        pts = pts + centroid  # move back
        return pts

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d

    def grid_3d_to_2d(self, grid_3d):
        b, h, w, _ = grid_3d.shape
        grid_2d = grid_3d / grid_3d[...,2:]
        grid_2d = grid_2d.matmul(self.K.to(grid_3d.device).transpose(2,1))[:,:,:,:2]
        WH = torch.FloatTensor([w-1, h-1]).to(grid_3d.device).view(1,1,1,2)
        grid_2d = grid_2d / WH *2.-1.  # normalize to -1~1
        return grid_2d

    def get_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat)
        grid_3d = self.translate_pts(grid_3d, self.trans_xyz)
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_inv_warped_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.translate_pts(grid_3d, -self.trans_xyz)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat.transpose(2,1))
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def get_inv_warped_2d_grid(self, depth):
        b, h, w = depth.shape
        grid_3d = self.get_inv_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def warp_canon_depth(self, canon_depth):
        b, h, w = canon_depth.shape
        grid_3d = self.get_warped_3d_grid(canon_depth).reshape(b,-1,3)
        faces = get_face_idx(b, h, w).to(canon_depth.device)
        # print(grid_3d.device, faces.device, self.R.device, self.t.device, self.K.device)
        warped_depth = self.renderer.render_depth(grid_3d, faces)

        # allow some margin out of valid range
        margin = (self.max_depth - self.min_depth) /2
        warped_depth = warped_depth.clamp(min=self.min_depth-margin, max=self.max_depth+margin)
        return warped_depth

    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = torch.FloatTensor([0,0,1]).to(depth.device)
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal

    def render_yaw(self, im, depth, v_before=None, v_after=None, rotations=None, maxr=90, nsample=9, crop_mesh=None):
        b, c, h, w = im.shape
        grid_3d = self.depth_to_3d_grid(depth)

        if crop_mesh is not None:
            top, bottom, left, right = crop_mesh  # pixels from border to be cropped
            if top > 0:
                grid_3d[:,:top,:,1] = grid_3d[:,top:top+1,:,1].repeat(1,top,1)
                grid_3d[:,:top,:,2] = grid_3d[:,top:top+1,:,2].repeat(1,top,1)
            if bottom > 0:
                grid_3d[:,-bottom:,:,1] = grid_3d[:,-bottom-1:-bottom,:,1].repeat(1,bottom,1)
                grid_3d[:,-bottom:,:,2] = grid_3d[:,-bottom-1:-bottom,:,2].repeat(1,bottom,1)
            if left > 0:
                grid_3d[:,:,:left,0] = grid_3d[:,:,left:left+1,0].repeat(1,1,left)
                grid_3d[:,:,:left,2] = grid_3d[:,:,left:left+1,2].repeat(1,1,left)
            if right > 0:
                grid_3d[:,:,-right:,0] = grid_3d[:,:,-right-1:-right,0].repeat(1,1,right)
                grid_3d[:,:,-right:,2] = grid_3d[:,:,-right-1:-right,2].repeat(1,1,right)

        grid_3d = grid_3d.reshape(b,-1,3)
        im_trans = []

        # inverse warp
        if v_before is not None:
            rot_mat, trans_xyz = get_transform_matrices(v_before)
            grid_3d = self.translate_pts(grid_3d, -trans_xyz)
            grid_3d = self.rotate_pts(grid_3d, rot_mat.transpose(2,1))

        if rotations is None:
            rotations = torch.linspace(-math.pi/180*maxr, math.pi/180*maxr, nsample)
        for i, ri in enumerate(rotations):
            ri = torch.FloatTensor([0, ri, 0]).to(im.device).view(1,3)
            rot_mat_i, _ = get_transform_matrices(ri)
            grid_3d_i = self.rotate_pts(grid_3d, rot_mat_i.repeat(b,1,1))

            if v_after is not None:
                if len(v_after.shape) == 3:
                    v_after_i = v_after[i]
                else:
                    v_after_i = v_after
                rot_mat, trans_xyz = get_transform_matrices(v_after_i)
                grid_3d_i = self.rotate_pts(grid_3d_i, rot_mat)
                grid_3d_i = self.translate_pts(grid_3d_i, trans_xyz)

            faces = get_face_idx(b, h, w).to(im.device)
            textures = get_textures_from_im(im, tx_size=self.tex_cube_size)
            warped_images = self.renderer.render_rgb(grid_3d_i, faces, textures).clamp(min=-1., max=1.)
            im_trans += [warped_images]
        return torch.stack(im_trans, 1)  # b x t x c x h x w
