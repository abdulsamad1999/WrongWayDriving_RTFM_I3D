import torch
import torch.nn as nn
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.FloatTensor')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(1, in_channels // 2)

        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d
        max_pool_layer = nn.MaxPool1d(kernel_size=2)

        self.g = conv_nd(self.in_channels, self.inter_channels, kernel_size=1)
        self.theta = conv_nd(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = conv_nd(self.in_channels, self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        return W_y + x


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels, inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d

        self.conv_1 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            bn(512)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv1d(512 * 3, 512, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)

        self.conv_5 = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            bn(512)
        )

        # ✅ Fix residual dimension mismatch with 1x1 conv
        self.residual_conv = nn.Conv1d(len_feature, 512, kernel_size=1)

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(x)
        out3 = self.conv_3(x)

        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out_d)
        out = self.non_local(out)
        out = torch.cat((out, out_d), dim=1)
        out = self.conv_5(out)

        residual = self.residual_conv(x)
        out = out + residual
        out = out.permute(0, 2, 1)
        return out


class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=n_features)

        # ✅ Fixed: FC now expects 512 input features after Aggregate
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        assert inputs.dim() == 3, f"Expected 3D input (batch, 32, 1024), got {inputs.shape}"

        out = inputs.permute(0, 2, 1)  # From (batch, 32, 1024) → (batch, 1024, 32)
        out = self.Aggregate(out)       # Output: (batch, 32, 512)
        out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))

        bs = inputs.size(0)
        ncrops = 1  # Single crop setup
        scores = scores.view(bs, ncrops, -1).mean(1).unsqueeze(2)

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)

        abnormal_features = features[self.batch_size:]
        abnormal_scores = scores[self.batch_size:]
        afea_magnitudes = feat_magnitudes[self.batch_size:]

        normal_features = features[:self.batch_size]
        normal_scores = scores[:self.batch_size]
        nfea_magnitudes = feat_magnitudes[:self.batch_size]

        n_size = nfea_magnitudes.shape[0]

        idx_abn = torch.topk(afea_magnitudes, self.k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand(-1, -1, abnormal_features.shape[2])

        idx_normal = torch.topk(nfea_magnitudes, self.k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand(-1, -1, normal_features.shape[2])

        total_select_abn_feature = torch.gather(abnormal_features, 1, idx_abn_feat)
        total_select_nor_feature = torch.gather(normal_features, 1, idx_normal_feat)

        idx_abn_score = idx_abn.unsqueeze(2).expand(-1, -1, abnormal_scores.shape[2])
        idx_normal_score = idx_normal.unsqueeze(2).expand(-1, -1, normal_scores.shape[2])

        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

        return score_abnormal, score_normal, total_select_nor_feature, total_select_abn_feature, \
               total_select_abn_feature, total_select_abn_feature, scores, \
               total_select_abn_feature, total_select_abn_feature, feat_magnitudes
