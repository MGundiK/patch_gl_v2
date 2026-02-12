import torch
from torch import nn


class InterPatchGating(nn.Module):
    """
    GLCN-inspired inter-patch gating module.
    
    Captures global dynamics by learning patch importance weights:
    GlobalAvgPool (over features) → MLP → Sigmoid → element-wise scaling.
    """
    def __init__(self, patch_num, reduction=4):
        super(InterPatchGating, self).__init__()
        hidden = max(patch_num // reduction, 2)
        self.mlp = nn.Sequential(
            nn.Linear(patch_num, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, patch_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B*C, patch_num, patch_len]
        w = x.mean(dim=2)          # GAP over feature dim → [B*C, patch_num]
        w = self.mlp(w)            # Importance weights   → [B*C, patch_num]
        w = w.unsqueeze(2)         # Broadcast shape      → [B*C, patch_num, 1]
        return x * w


class MultiscaleLocalConv(nn.Module):
    """
    Lighter multiscale local feature extraction with kernels {1, 3, 5}.
    
    v2 changes vs v1:
    - Reduced from {1,3,5,7} to {1,3,5} — fewer params, less overfitting
    - Added dropout before output for regularization
    """
    def __init__(self, patch_num, kernels=(1, 3, 5), dropout=0.1):
        super(MultiscaleLocalConv, self).__init__()
        self.n_scales = len(kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=patch_num,
                out_channels=patch_num,
                kernel_size=k,
                padding=k // 2,
                groups=patch_num
            )
            for k in kernels
        ])
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(patch_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*C, patch_num, patch_len]
        out = sum(conv(x) for conv in self.convs) / self.n_scales
        out = self.gelu(out)
        out = self.bn(out)
        out = self.dropout(out)
        return out


class GLPatchNetwork(nn.Module):
    """
    GLPatch v2 dual-stream network.
    
    v2 changes vs v1:
    1. REMOVED aggregate Conv1D — it was smoothing the seasonality residual,
       blurring the sharp patterns the non-linear stream needs to capture.
    2. LIGHTER multiscale conv — kernels {1,3,5} instead of {1,3,5,7}, with dropout.
    3. LEARNABLE residual scaling — alpha parameter lets the model learn how
       much weight to give the new modules vs passing through unchanged.
       Initialized at 0.1 so training starts close to xPatch behavior.
    4. KEPT inter-patch gating — demonstrably helps at longer horizons.
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(GLPatchNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality)
        # ================================================================

        # Patching (overlapping, from xPatch)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding (from xPatch)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise (from xPatch)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream (from xPatch)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # [GLCN] Inter-patch Gating: learns global patch importance
        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)

        # [GLCN] Multiscale Local Conv: multi-resolution local features
        self.multiscale_conv = MultiscaleLocalConv(self.patch_num,
                                                   kernels=(1, 3, 5),
                                                   dropout=0.1)

        # [v2] Learnable residual scaling — initialized small so the model
        # starts close to vanilla xPatch and gradually learns to use the new modules
        self.res_alpha = nn.Parameter(torch.tensor(0.1))

        # CNN Pointwise (from xPatch)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head (from xPatch)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Linear Stream (Trend) — identical to xPatch
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # Streams Concatenation
        # ================================================================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # s: seasonality [Batch, Input, Channel]
        # t: trend       [Batch, Input, Channel]

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))  # [B*C, I]
        t = torch.reshape(t, (B * C, I))  # [B*C, I]

        # ---- Non-linear Stream ----

        # Patching (overlapping, from xPatch)
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [B*C, patch_num, patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res
        # s: [B*C, patch_num, patch_len]

        # [GLCN modules with learnable residual scaling]
        s_base = s  # save baseline (xPatch-equivalent output)

        s = self.inter_patch_gate(s)        # global patch importance
        s = self.multiscale_conv(s)         # multi-resolution local features

        s = s_base + self.res_alpha * s     # scaled residual

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # ---- Linear Stream (identical to xPatch) ----

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # ---- Streams Concatenation ----
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
