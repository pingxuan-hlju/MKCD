import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5, 
        spline_order=3, 
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0, 
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU, 
        grid_eps=0.02, 
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        ) 
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        ) 
        B = y.transpose(0, 1)  
        solution = torch.linalg.lstsq(
            A, B
        ).solution  
        result = solution.permute(
            2, 0, 1
        ) 

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x) 
        splines = splines.permute(1, 0, 2) 
        orig_coeff = self.scaled_spline_weight 
        orig_coeff = orig_coeff.permute(1, 2, 0)  
        unreduced_spline_output = torch.bmm(splines, orig_coeff) 
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        ) 

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_linear = nn.Linear(d_model, self.d_k * num_heads)
        self.k_linear = nn.Linear(d_model, self.d_k * num_heads)
        self.v_linear = nn.Linear(d_model, self.d_v * num_heads)
        self.out = nn.Linear(self.d_v * num_heads, d_model)
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)
        nn.init.constant_(self.v_linear.bias, 0)
        nn.init.constant_(self.out.bias, 0)
        
    def forward(self, H, rwr, device):
        Q = self.q_linear(H).view(-1, self.num_heads, self.d_k)
        K = self.k_linear(H).view(-1, self.num_heads, self.d_k)
        V = self.v_linear(H).view(-1, self.num_heads, self.d_v)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        rg = rwr.rg.unsqueeze(0).repeat(scores.shape[0], 1, 1).to(device)
        scores = scores * rg
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        context = context.transpose(0, 1).contiguous().view(-1, self.d_v * self.num_heads)
        out = self.out(context)
        return out
    
class Gate_GraphTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(Gate_GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([MultiHeadSelfAttention(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.eta_1 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.eta_2 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.eta_3 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.weight = [self.eta_1, self.eta_2, self.eta_3]
        self.gate = nn.Linear(1527 * 2, 1527)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for layer in self.layers:
            layer._reset_parameters()
        nn.init.constant_(self.eta_1, 1)
        nn.init.constant_(self.eta_2, 1)
        nn.init.constant_(self.eta_3, 1)
        
    def forward(self, feature, faeture, device):
        rwr = combine_rwr(feature, weights=self.weight)
        features = feature
        for layer in self.layers:
            features = features + layer(features, rwr,device)
        gate_input = torch.cat([features, feature], dim=1)
        gate_output = torch.sigmoid(self.gate(gate_input))
        feature = (1 - gate_output) * feature + gate_output * faeture
        return feature

class Data:
    def __init__(self, rg: torch.Tensor, sigama: float):
        self.rg = rg.float()
        self.sigama = sigama

def line_normalization(a: torch.Tensor) -> torch.Tensor:
    an = a.clone()
    s = a.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    return an / s

def rand_walk(alpha: float, steps: int, A: torch.Tensor) -> list:
    P = A.clone()
    P.fill_diagonal_(0)
    P = line_normalization(P)
    E = A
    ans = [E]
    W = E
    for s in range(steps):
        W = (1 - alpha) * torch.matmul(P.T, W) + alpha * E
        ans.append(W.T)
    return ans

def combine_rwr(A: torch.Tensor, weights: list, ) -> torch.Tensor:
    rwr = rand_walk(0.7, 2, A)
    sigama = 2e-3
    rg = sum([w * weight for w, weight in zip(rwr, weights)])
    return Data(rg=rg, sigama=sigama)

class Net(nn.Module):
    def __init__(self, in_dim=1527):
        super(Net,self).__init__()
        self.transformer = Gate_GraphTransformer(d_model=in_dim, num_heads=4, num_layers=2)
        self.c1 = nn.Conv2d(1, 32, kernel_size=(2,2), stride=1, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.c2 = nn.Conv2d(32, 64, kernel_size=(1, 2), stride=1, padding=0) 
        self.c3 = nn.Conv2d(64, 128, kernel_size=(1, 2), stride=1, padding=0) 
        self.p2 = nn.MaxPool2d(kernel_size=(1, 7)) 
        self.l1 = nn.Linear(1920, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 2) 
        self.resKan = KAN([1527*2, 1024, 256])
        # self.multkan = MultKAN([3968, 512, 2]) # MultKAN模型
        self.leakyrelu = nn.LeakyReLU()
        self.d = nn.Dropout(0.5) 
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        
    #权重参数初始化
    def reset_para(self):
        nn.init.xavier_normal_(self.c1.weight)
        nn.init.xavier_normal_(self.c2.weight) 
        nn.init.xavier_normal_(self.c3.weight)
        nn.init.xavier_normal_(self.l1.weight, gain= nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.l2.weight, gain= nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.l3.weight) 
        nn.init.xavier_normal_(self.gate.weight) 
        torch.nn.init.constant_(self.gate.bias) 
        
    def forward(self, x1, x2, features, device):
        feature = self.transformer(features, features, device)
        x2 = x2 + 834
        feature = torch.cat([feature[x1][:,None,None,:], feature[x2][:,None,None,:]], dim=2) 
        resKan_x = feature
        resKan_x = resKan_x.reshape(resKan_x.shape[0], -1)
        resKan_x = self.resKan(resKan_x)
        x = feature
        x = self.leakyrelu(self.c1(x))
        x = self.p1(x)
        x = self.leakyrelu(self.c2(x))
        x = self.p2(x)
        x = self.leakyrelu(self.c3(x))
        x = self.p2(x)
        alpha_beta = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        self.alpha.data = alpha_beta[0].data
        self.beta.data = alpha_beta[1].data
        x = x.reshape(x.shape[0], -1)
        x = self.leakyrelu(self.l1(x))
        x = self.d(x)
        x = self.leakyrelu(self.l2(x))
        x = self.alpha * x + self.beta * resKan_x
        x = self.d(x)
        x = self.l3(x)
        return x 
