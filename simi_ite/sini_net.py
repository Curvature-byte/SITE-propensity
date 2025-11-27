import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 辅助函数：安全开方
def safe_sqrt(x, lbound=1e-10):
    return torch.sqrt(torch.clamp(x, min=lbound))

class PDDMNet(nn.Module):
    """
    PDDM 子网络：用于计算两点之间的相似度
    对应原代码中的 self.pddm(...)
    """
    def __init__(self, dim_in, dim_pddm, dim_c, dim_s, act_func):
        super(PDDMNet, self).__init__()
        self.act = act_func
        
        # 定义 PDDM 的权重
        # u 分支: |x_i - x_j|
        self.w_u = nn.Linear(dim_in, dim_pddm)
        # v 分支: (x_i + x_j) / 2
        self.w_v = nn.Linear(dim_in, dim_pddm)
        
        # Concatenate 分支
        self.w_c = nn.Linear(dim_pddm * 2, dim_c)
        
        # Score 分支
        self.w_s = nn.Linear(dim_c, dim_s)

    def forward(self, x_i, x_j):
        # x_i, x_j shape: [1, dim_in] or [batch, dim_in]
        
        u = torch.abs(x_i - x_j)
        v = (x_i + x_j) / 2.0
        
        u_embed = self.act(self.w_u(u))
        v_embed = self.act(self.w_v(v))
        
        # L2 Normalize (对应原代码 dim=0, 但在Batch中通常 dim=1，此处保持逻辑对单样本特征归一化)
        # 注意：原代码 slice 出来是 [1, dim]，dim=0 归一化其实没有意义（结果全是1）。
        # 假设原意是归一化特征向量，PyTorch 中用 dim=1
        u_embed = F.normalize(u_embed, p=2, dim=1)
        v_embed = F.normalize(v_embed, p=2, dim=1)
        
        c = torch.cat([u_embed, v_embed], dim=1)
        c = self.act(self.w_c(c))
        s = self.w_s(c)
        return s

class SiteNet(nn.Module):
    def __init__(self, dims, FLAGS):
        super(SiteNet, self).__init__()
        self.FLAGS = FLAGS
        
        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_pddm = dims[3]
        dim_c = dims[4]
        dim_s = dims[5]
        
        # 1. 激活函数选择
        if FLAGS.nonlin.lower() == 'elu':
            self.act = F.elu
        else:
            self.act = F.relu
            
        # 2. 构建表示层 (Representation Layers)
        self.rep_layers = nn.ModuleList()
        self.rep_bns = nn.ModuleList() # 存储 BatchNorm
        
        # 确定第一层维度和逻辑
        current_dim = dim_input
        input_layer_idx = 0
        
        # 变量选择层 (Variable Selection)
        if FLAGS.varsel:
            # 第一层是逐元素相乘 (Rescaling)，不是全连接
            self.varsel_weight = nn.Parameter(torch.ones(1, dim_input) / dim_input)
            input_layer_idx = 1 # 实际上第一层循环被这个替代了
        else:
            self.varsel_weight = None

        # 构建后续的全连接层
        for i in range(input_layer_idx, FLAGS.n_in):
            # 确定输入输出维度
            d_in = dim_input if (i == 0 and not FLAGS.varsel) else dim_in
            d_out = dim_in
            
            self.rep_layers.append(nn.Linear(d_in, d_out))
            
            if FLAGS.batch_norm:
                self.rep_bns.append(nn.BatchNorm1d(d_out))
            else:
                self.rep_bns.append(None)
                
        # 3. 构建 PDDM 单元 (权重共享)
        self.pddm_net = PDDMNet(dim_in, dim_pddm, dim_c, dim_s, self.act)
        
        # 4. 构建输出层 (Output Layers)
        # 根据 FLAGS.split_output 决定是 TARNet 结构 (2个头) 还是 S-Learner (1个头)
        self.head_treated = nn.ModuleList()
        self.head_control = nn.ModuleList() # 仅在 split_output=True 时使用
        self.pred_treated = None
        self.pred_control = None
        
        dim_out_input = dim_in
        if not FLAGS.split_output:
            # 如果不拆分，输入需要拼接 t，维度 + 1
            dim_out_input = dim_in + 1
            
        # 定义构建 Output Head 的辅助函数
        def build_head(input_d):
            layers = nn.ModuleList()
            curr = input_d
            # 中间层
            for _ in range(FLAGS.n_out):
                # 假设 dim_out 在中间层也适用，原代码逻辑：dims = [dim_in] + ([dim_out]*FLAGS.n_out)
                layers.append(nn.Linear(curr, dim_out))
                curr = dim_out
            # 最终预测层
            pred_layer = nn.Linear(curr, 1) 
            return layers, pred_layer

        if FLAGS.split_output:
            self.head_layers_1, self.head_pred_1 = build_head(dim_in) # Treated
            self.head_layers_0, self.head_pred_0 = build_head(dim_in) # Control
        else:
            self.head_layers_common, self.head_pred_common = build_head(dim_out_input)

    def forward(self, x, t, three_pairs=None):
        """
        :param x: Covariates [batch, dim_input]
        :param t: Treatment [batch, 1]
        :param three_pairs: PDDM 专用的输入数据 [6, dim_input] (batch size 固定为 6)
        """
        FLAGS = self.FLAGS
        
        # ===========================
        # Part 1: Representation Learning
        # ===========================
        
        # 统一处理 x 和 three_pairs (如果有)
        # 为了复用逻辑，我们可以把它们拼起来过网络，然后再拆开
        if three_pairs is not None and FLAGS.p_pddm > 0:
            x_all = torch.cat([x, three_pairs], dim=0)
            batch_split_idx = x.shape[0]
        else:
            x_all = x
            batch_split_idx = None

        h = x_all
        
        # 变量选择 (Variable Selection)
        if FLAGS.varsel:
            # 广播乘法: [N, D] * [1, D]
            h = h * self.varsel_weight
            
        # 深度网络层
        for i, layer in enumerate(self.rep_layers):
            h = layer(h)
            
            # Batch Norm
            if self.rep_bns[i] is not None:
                h = self.rep_bns[i](h)
            
            # Activation
            h = self.act(h)
            
            # Dropout
            h = F.dropout(h, p=self.FLAGS.do_in, training=self.training)
            
        h_rep = h # 未归一化的表示
        
        # 归一化 (Normalization)
        if FLAGS.normalization == 'divide':
            # dim=1 是特征维度
            h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep**2, dim=1, keepdim=True))
        else:
            h_rep_norm = h_rep

        # 拆分回 x 和 three_pairs
        if batch_split_idx is not None:
            rep_x = h_rep_norm[:batch_split_idx]
            rep_pairs = h_rep_norm[batch_split_idx:]
        else:
            rep_x = h_rep_norm
            rep_pairs = None
            
        # ===========================
        # Part 2: Output Heads
        # ===========================
        y_pred = torch.zeros_like(t) # [N, 1]
        
        if FLAGS.split_output:
            # Split Output (TARNet style)
            # Control Head
            h0 = rep_x
            for layer in self.head_layers_0:
                h0 = self.act(layer(h0))
                h0 = F.dropout(h0, p=FLAGS.do_out, training=self.training)
            y0 = self.head_pred_0(h0)
            
            # Treated Head
            h1 = rep_x
            for layer in self.head_layers_1:
                h1 = self.act(layer(h1))
                h1 = F.dropout(h1, p=FLAGS.do_out, training=self.training)
            y1 = self.head_pred_1(h1)
            
            # Dynamic Stitch (利用 mask 选择)
            # t should be 0 or 1
            y_pred = y0 * (1 - t) + y1 * t
            
        else:
            # S-Learner style (Concat t)
            h = torch.cat([rep_x, t], dim=1)
            for layer in self.head_layers_common:
                h = self.act(layer(h))
                h = F.dropout(h, p=FLAGS.do_out, training=self.training)
            y_pred = self.head_pred_common(h)

        return y_pred, rep_x, rep_pairs

    def calculate_loss(self, y_pred, y_true, t, p_t, rep_pairs, three_pairs_simi, 
                       r_lambda, r_mid_point_mini, r_pddm):
        """
        计算所有 Loss
        """
        FLAGS = self.FLAGS
        
        # 1. Sample Reweighting
        if FLAGS.reweight_sample:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        # 2. Factual Loss (Risk)
        if FLAGS.loss == 'l1':
            res = torch.abs(y_true - y_pred)
            risk = torch.mean(sample_weight * res)
            pred_error = torch.mean(res) # simple mean for report
        elif FLAGS.loss == 'log':
            # Log loss 需要 y_pred 是 logits 还是 probability? 
            # 原代码: y = sigmoid... res = y_*log(y)...
            # 这里建议 y_pred 输出 logits，然后用 BCEWithLogitsLoss，但为了对其原代码逻辑：
            y_prob = 0.995 / (1.0 + torch.exp(-y_pred)) + 0.0025
            res = y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob)
            risk = -torch.mean(sample_weight * res)
            pred_error = -torch.mean(res)
        else: # L2 / MSE
            res = torch.square(y_true - y_pred)
            risk = torch.mean(sample_weight * res)
            pred_error = torch.sqrt(torch.mean(res))

        tot_error = risk
        
        # 3. Regularization (Weight Decay)
        # 手动计算 L2 loss 以匹配原代码的逻辑 (只针对特定层)
        wd_loss = 0.0
        if FLAGS.p_lambda > 0 and FLAGS.rep_weight_decay:
            # 针对 Representation Layers
            for i, layer in enumerate(self.rep_layers):
                if not (FLAGS.varsel and i == 0): # 变量选择层通常不加 L2
                    wd_loss += torch.sum(layer.weight ** 2) / 2.0
            
            # 针对 Output Layers 的 w_pred
            if FLAGS.split_output:
                # dim_out-1 切片逻辑在 pytorch 中通常直接对 weight 做
                # 这里简单处理，对最后一层全连接加正则
                wd_loss += torch.sum(self.head_pred_0.weight ** 2) / 2.0
                wd_loss += torch.sum(self.head_pred_1.weight ** 2) / 2.0
            else:
                 # 原代码排除 Treatment coefficient? 
                 # tf.slice(weights_pred,[0,0],[dim_out-1,1])
                 w = self.head_pred_common.weight # shape [1, dim_out]
                 if w.shape[1] > 1:
                     wd_loss += torch.sum(w[:, :-1] ** 2) / 2.0
                 else:
                     wd_loss += torch.sum(w ** 2) / 2.0

            tot_error = tot_error + r_lambda * wd_loss

        # 4. PDDM Loss
        pddm_loss_val = torch.tensor(0.0, device=y_pred.device)
        mid_dist_val = torch.tensor(0.0, device=y_pred.device)

        if rep_pairs is not None and FLAGS.p_pddm > 0:
            # rep_pairs shape: [6, dim_in]
            # 索引映射: 0:i, 1:j, 2:k, 3:l, 4:m, 5:n
            x_i = rep_pairs[0:1]
            x_j = rep_pairs[1:2]
            x_k = rep_pairs[2:3]
            x_l = rep_pairs[3:4]
            x_m = rep_pairs[4:5]
            x_n = rep_pairs[5:6]
            
            # 计算 PDDM Score
            s_kl = self.pddm_net(x_k, x_l)
            s_mn = self.pddm_net(x_m, x_n)
            s_km = self.pddm_net(x_k, x_m)
            s_ik = self.pddm_net(x_i, x_k)
            s_jm = self.pddm_net(x_j, x_m)
            
            # 获取 Ground Truth Similarity
            # three_pairs_simi shape 预期为 [5, 1]
            simi_kl = three_pairs_simi[0:1]
            simi_mn = three_pairs_simi[1:2]
            simi_km = three_pairs_simi[2:3]
            simi_ik = three_pairs_simi[3:4]
            simi_jm = three_pairs_simi[4:5]
            
            pddm_loss_val = torch.sum(
                (simi_kl - s_kl)**2 + 
                (simi_mn - s_mn)**2 + 
                (simi_km - s_km)**2 + 
                (simi_ik - s_ik)**2 + 
                (simi_jm - s_jm)**2
            )
            
            tot_error = tot_error + r_pddm * pddm_loss_val
            
            # 5. Mid Point Distance Minimization
            if FLAGS.p_mid_point_mini > 0:
                mid_jk = (x_j + x_k) / 2.0
                mid_im = (x_i + x_m) / 2.0
                mid_dist_val = torch.sum((mid_jk - mid_im)**2)
                
                tot_error = tot_error + r_mid_point_mini * mid_dist_val

        return tot_error, pred_error, pddm_loss_val, mid_dist_val