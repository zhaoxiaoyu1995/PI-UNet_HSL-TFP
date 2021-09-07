import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class OutsideLoss(_Loss):
    def __init__(
        self, base_loss=MSELoss(reduction='mean'), length=0.1, u_D=298, bcs=None, nx=21
    ):
        super().__init__()
        self.base_loss = base_loss
        self.u_D = u_D
        self.bcs = bcs
        self.nx = nx
        self.length = length

    def forward(self, x):
        N, C, W, H = x.shape
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all bcs are Dirichlet
            d1 = x[:, :, :1, :]
            d2 = x[:, :, -1:, :]
            d3 = x[:, :, 1:-1, :1]
            d4 = x[:, :, 1:-1, -1:]
            point = torch.cat([d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten()], dim=0)
            return self.base_loss(point, torch.ones_like(point) * 0)
        loss = 0
        loss_consistency = 0
        for bc in self.bcs:
            if bc[0][1] == 0 and bc[1][1] == 0:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, :1]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][1] == self.length and bc[1][1] == self.length:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, -1:]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == 0 and bc[1][0] == 0:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., :1, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == self.length and bc[1][0] == self.length:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., -1:, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            else:
                raise ValueError("bc error!")
        return loss


class LaplaceLoss(_Loss):
    def __init__(
        self, base_loss=MSELoss(reduction='mean'), nx=21,
        length=0.1, weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], bcs=None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight = torch.Tensor(weight)
        self.bcs = bcs
        self.length = length
        self.nx = nx
        self.scale_factor = 1 # self.nx/200
        TEMPER_COEFFICIENT = 1 # 50.0
        STRIDE = self.length / self.nx
        self.cof = -1 * STRIDE**2/TEMPER_COEFFICIENT

    def laplace(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat):
        layout = interpolate(layout, scale_factor=self.scale_factor)

        heat = pad(heat, [1, 1, 1, 1], mode='reflect')    # constant, reflect, reflect
        layout_pred = self.laplace(heat)
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            return self.base_loss(layout_pred[..., 1:-1, 1:-1], self.cof * layout[..., 1:-1, 1:-1])
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length) + 1
                    layout_pred[..., idx_start:idx_end, :1] = self.cof * layout[..., idx_start:idx_end, :1]
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, -1:] = self.cof * layout[..., idx_start:idx_end, -1:]
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., :1, idx_start:idx_end] = self.cof * layout[..., :1, idx_start:idx_end]
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., -1:, idx_start:idx_end] = self.cof * layout[..., -1:, idx_start:idx_end]
                else:
                    raise ValueError("bc error!")
        return self.base_loss(layout_pred, self.cof * layout)


class Jacobi_layer(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, bcs=None
    ):
        super(Jacobi_layer, self).__init__()
        self.length = length
        self.bcs = bcs
        # 雅克比迭代的权重 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # 填充
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1 # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter):
        # 右端项
        f = self.cof * layout
        # G: 是否为非边界点
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat * G, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x


class Jacobi_layerSoft(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, bcs=None
    ):
        super(Jacobi_layerSoft, self).__init__()
        self.length = length
        self.bcs = bcs
        # 雅克比迭代的权重 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # 填充
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1 # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter):
        # 右端项
        f = self.cof * layout
        # G: 是否为非边界点
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x


class OHEMF12d(torch.nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, loss_fun, weight=None):
        super(OHEMF12d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))


def dfdx(f, dydeta, dydxi, Jinv, h):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, Jinv, h):
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


class Laplace_ununiform(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, batch_size=32, bcs=None, xs=None, ys=None
    ):
        super(Laplace_ununiform, self).__init__()
        self.length = length
        self.bcs = bcs
        self.nx = nx
        self.cof = 1.0
        assert (xs is not None) and (ys is not None)
        w, a_m, a_p, b_m, b_p = self.pre_coef(xs, ys)
        n_x, n_y = xs.shape
        self.w = w.expand(1, 1, n_x, n_y).repeat(batch_size, 1, 1, 1)
        self.a_m = a_m.expand(1, 1, n_x, n_y).repeat(batch_size, 1, 1, 1)
        self.a_p = a_p.expand(1, 1, n_x, n_y).repeat(batch_size, 1, 1, 1)
        self.b_m = b_m.expand(1, 1, n_x, n_y).repeat(batch_size, 1, 1, 1)
        self.b_p = b_p.expand(1, 1, n_x, n_y).repeat(batch_size, 1, 1, 1)

    def pre_hx(self, xs):
        hx = xs[1:, :] - xs[0:-1, :]
        hx_id1 = torch.cat((hx[0:1, :], hx), dim=0)
        hx_i = torch.cat((hx, hx[-1:, :]), dim=0)
        return hx_id1, hx_i

    def pre_hy(self, ys):
        hy = ys[:, 1:] - ys[:, 0:-1]
        hy_jd1 = torch.cat((hy[:, 0:1], hy), dim=1)
        hy_j = torch.cat((hy, hy[:, -1:]), dim=1)
        return hy_jd1, hy_j

    def pre_coef(self, xs, ys):
        hx_id1, hx_i = self.pre_hx(xs)
        hy_jd1, hy_j = self.pre_hy(ys)
        w = (2.0 / (hx_i + hx_id1)) * (-1.0 / hx_i + (-1) / hx_id1) + \
            (2.0 / (hy_j + hy_jd1)) * (-1.0 / hy_j + (-1) / hy_jd1)
        a_m = (2.0 / (hx_i + hx_id1)) * (1.0 / hx_id1)
        a_p = (2.0 / (hx_i + hx_id1)) * (1.0 / hx_i)
        b_m = (2.0 / (hy_j + hy_jd1)) * (1.0 / hy_jd1)
        b_p = (2.0 / (hy_j + hy_jd1)) * (1.0 / hy_j)
        return w, a_m, a_p, b_m, b_p

    def jacobi(self, x):
        # x_f, x_b, x_u, x_d represents the elements of the direction of forward, back, up, down
        x_f = torch.cat((x[:, :, :, 1:], x[:, :, :, -2:-1]), dim=3).detach()
        x_b = torch.cat((x[:, :, :, 1:2], x[:, :, :, 0:-1]), dim=3).detach()
        x_u = torch.cat((x[:, :, 1:2, :], x[:, :, 0:-1, :]), dim=2).detach()
        x_d = torch.cat((x[:, :, 1:, :], x[:, :, -2:-1, :]), dim=2).detach()
        # y = self.w * x + self.a_m * x_b + self.a_p * x_f + self.b_m * x_u + self.b_p * x_d
        y = self.w * x + self.a_m * x_u + self.a_p * x_d + self.b_m * x_b + self.b_p * x_f
        return y

    def forward(self, layout, heat):
        # 右端项
        f = self.cof * layout
        # G: 是否为非边界点
        G = torch.ones_like(heat)

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    idx_start = 79
                    idx_end = 121
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")

        loss = G * (self.jacobi(G * heat) + f)
        return loss


class heat_conservation(torch.nn.Module):
    def __init__(self, nx=21, length=0.1, bcs=None):
        super(heat_conservation, self).__init__()
        self.length = length
        self.bcs = bcs
        # 雅克比迭代的权重 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # 填充
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

        # different scale heat conservation
        self.scale_4_weight = torch.ones(4, 4).view(1, 1, 4, 4)
        self.scale_10_weight = torch.ones(10, 10).view(1, 1, 10, 10)
        self.scale_50_weight = torch.ones(50, 50).view(1, 1, 50, 50)
        self.scale_200_weight = torch.ones(200, 200).view(1, 1, 200, 200)

        # Define loss function
        self.loss = OHEMF12d(loss_fun=F.l1_loss)
        # self.loss = torch.nn.L1Loss()

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def scale_conservation(self, x, weight, stride=1):
        return conv2d(x, weight.to(device=x.device), bias=None, stride=stride, padding=0)

    def forward(self, layout, heat):
        # 右端项
        f = self.cof * layout
        # G: 是否为非边界点
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        with torch.no_grad():
            x = F.pad(heat.detach() * G, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)

        vertical_direction = torch.mean(x, dim=2)
        horizontal_diretion = torch.mean(x, dim=3)
        return self.loss(vertical_direction, torch.mean(heat * G, dim=2)) + \
               self.loss(horizontal_diretion, torch.mean(heat * G, dim=3))


class MultiScaleLayout:
    def __init__(self):
        # Different scale weight
        self.scale_8_weight = torch.ones(8, 8).view(1, 1, 8, 8) / 64.0
        self.scale_4_weight = torch.ones(4, 4).view(1, 1, 4, 4) / 16.0
        self.scale_2_weight = torch.ones(2, 2).view(1, 1, 2, 2) / 4.0

    def conv(self, x, weight, stride=1):
        return conv2d(x, weight.to(device=x.device), bias=None, stride=stride, padding=0)

    def __call__(self, layout):
        layout_scale_8 = self.conv(layout, self.scale_8_weight, stride=8)
        layout_scale_4 = self.conv(layout, self.scale_4_weight, stride=4)
        layout_scale_2 = self.conv(layout, self.scale_2_weight, stride=2)
        return layout_scale_8, layout_scale_4, layout_scale_2, layout