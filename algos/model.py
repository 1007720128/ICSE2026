import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, SAGPooling, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import dense_to_sparse, to_dense_adj

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def orthogonal_initializer(shape, dtype=np.float32):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape).astype(dtype)


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, n_actions, n_mcs=5, dim_act=3):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.n_action = n_actions
        obs_dim = dim_observation * n_agent
        act_dim = n_agent * n_mcs * dim_act

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)
        self.param_init()

    def param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.elu_(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.elu_(self.FC2(combined))
        return self.FC4(F.elu_(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, n_mcs, n_actions=0, agent=0, adj=None, n_heads=3,
                 n_hidden_gat=100, n_out_gat=100):
        """
        初始化函数
        Args:
            n_hidden_gat:
            n_out_gat:
            n_mcs: 微服务数量
            n_actions: 动作数
            agent: 服务器数量
            adj: 微服务的邻接矩阵
            n_heads: 多头注意力的头数
        """
        super(Actor, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(3 * n_mcs + 2, 100), nn.ELU())
        self.dense2 = nn.Sequential(nn.Linear(100 + n_heads * n_mcs * n_out_gat, 128),
                                    nn.ELU())
        self.dense_outs = nn.ModuleList([nn.Sequential(nn.Linear(128, n_actions),
                                                       nn.ELU()) for _ in range(n_mcs)])
        # self.param_init()
        self.agent = agent
        self.adj = adj
        self.gat = GAT(2 * agent + 2)
        self.add_module("gat", self.gat)
        self.n_mcs = n_mcs

        self.param_init()

    def param_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0, std=0.7)

    def forward(self, obs):
        s = self.agent
        m = self.n_mcs

        server_state = obs[:, :3 * m + 2]
        mcs_res = torch.reshape(obs[:, 3 * m + 2:4 * m + 2], (-1, m, 1))
        mcs_ins = torch.reshape(obs[:, 4 * m + 2:5 * m + 2], (-1, m, 1))
        mcs_resp_server = torch.reshape(obs[:, 5 * m + 2:5 * m + 2 + m * s], (-1, m, s))
        mcs_insp_server = torch.reshape(obs[:, 5 * m + 2 + m * s:], (-1, m, s))

        # 服务器图特征(m, f)
        mcs_feat = torch.concat([mcs_res, mcs_ins, mcs_resp_server, mcs_insp_server],
                                dim=-1)
        # (batch, m, n_heads * 80)
        mcs_gat_feat = self.gat(mcs_feat, self.adj)
        mcs_gat_feat = torch.reshape(mcs_gat_feat, (mcs_gat_feat.size()[0], -1))
        # (batch, 500)
        server_feat = F.relu(self.dense1(server_state))
        server_feat = torch.reshape(server_feat, (server_feat.size()[0], -1))

        # (batch, 500 + m * n_heads * self.n_out_gat)
        hidden_feat = F.relu(self.dense2(torch.cat([server_feat, mcs_gat_feat], dim=-1)))
        actions = []
        for i in range(self.n_mcs):
            actions.append(F.gumbel_softmax(F.tanh(self.dense_outs[i](hidden_feat)),
                                            tau=1, dim=-1, hard=True))
        # actions: (batch, n_mcs x n_acitons)
        actions = torch.cat(actions, dim=-1)
        return actions


class PoolCritic(nn.Module):
    def __init__(self, n_agent, dim_observation, n_actions, n_mcs=5, dim_act=3):
        super(PoolCritic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.n_action = n_actions
        obs_dim = dim_observation * n_agent
        act_dim = n_agent * n_mcs

        self.FC1 = nn.Sequential(nn.Linear(obs_dim, 220))
        self.FC2 = nn.Sequential(nn.Linear(220 + act_dim, 170))
        self.FC3 = nn.Sequential(nn.Linear(170, 50))
        self.FC4 = nn.Sequential(nn.Linear(50, 1))
        self.param_init()

    def param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = self.FC1(obs)
        combined = torch.cat([result, acts], 1)
        result = self.FC2(combined)
        return self.FC4(self.FC3(result))


class PoolActor(nn.Module):
    def __init__(self, n_mcs, agent=0, adj=None, n_heads=3, n_out_gat=100, L=5, ratio=1):
        """
        初始化函数
        Args:
            n_hidden_gat:
            n_out_gat:
            n_mcs: 微服务数量
            n_actions: 动作数
            agent: 服务器数量
            adj: 微服务的邻接矩阵
            n_heads: 多头注意力的头数
        """
        super(PoolActor, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(3 * n_mcs + 2, 100), nn.ELU())
        self.dense2 = nn.Sequential(nn.Linear(100 + n_heads * math.ceil(n_mcs * ratio) * n_out_gat + L * 2 * 150, 128),
                                    nn.ELU())
        self.dense_outs = nn.ModuleList([nn.Sequential(nn.Linear(128, 1), nn.LeakyReLU()) for _ in range(n_mcs)])
        # self.param_init()
        self.agent = agent
        self.adj = adj
        # self.gat2conv = GATv2Conv(in_channels=2 * agent + 2, out_channels=n_out_gat, heads=n_heads)
        # self.pooling = TopKPooling(in_channels=n_heads * n_out_gat, ratio=ratio)

        self.gat = GAT(2 * agent + 2, n_out=n_out_gat)
        self.bilstm = nn.LSTM(input_size=77 + n_mcs, hidden_size=150, bidirectional=True, num_layers=1,
                              batch_first=True)
        # self.add_module("gat", self.gat)
        self.n_mcs = n_mcs

        self.param_init()

    def param_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0, std=0.7)

    def forward(self, obs, past_obs):
        # obs: (B,F) past_obs: (B,L,F)
        s = self.agent
        m = self.n_mcs
        edge_index, _ = dense_to_sparse(self.adj)

        # past_enc: (B,L,2*F)
        past_enc, (h, c) = self.bilstm(past_obs)
        past_enc = torch.reshape(past_enc, (past_enc.size()[0], -1))
        server_state = obs[:, :3 * m + 2]
        mcs_res = torch.reshape(obs[:, 3 * m + 2:4 * m + 2], (-1, m, 1))
        mcs_ins = torch.reshape(obs[:, 4 * m + 2:5 * m + 2], (-1, m, 1))
        mcs_resp_server = torch.reshape(obs[:, 5 * m + 2:5 * m + 2 + m * s], (-1, m, s))
        mcs_insp_server = torch.reshape(obs[:, 5 * m + 2 + m * s:], (-1, m, s))

        # 服务器图特征(B, m, f)
        mcs_feat = torch.concat([mcs_res, mcs_ins, mcs_resp_server, mcs_insp_server],
                                dim=-1)

        # (batch, m, n_heads * 80)
        batch_feats = []
        gat_feat = self.gat(mcs_feat, self.adj)
        # for i in range(gat_feat.size(0)):
        #     pooled_gat_feat = self.pooling(gat_feat[i], edge_index=edge_index)
        #     batch_feats.append(pooled_gat_feat[0])
        # mcs_gat_feat = torch.stack(batch_feats)
        # # (B, (m/2)*F) 向上取整
        # mcs_gat_feat = torch.reshape(mcs_gat_feat, (mcs_gat_feat.size(0), -1))
        mcs_gat_feat = torch.reshape(gat_feat, (gat_feat.size(0), -1))

        # (batch, 500)
        server_feat = self.dense1(server_state)
        server_feat = torch.reshape(server_feat, (server_feat.size(0), -1))

        # (batch, 500 + m * n_heads * self.n_out_gat)
        hidden_feat = self.dense2(torch.cat([server_feat, mcs_gat_feat, past_enc], dim=-1))
        actions = []
        for i in range(self.n_mcs):
            actions.append(self.dense_outs[i](hidden_feat))
        # actions: (batch, n_mcs)
        actions = torch.cat(actions, dim=-1)
        return actions


class PoolActorV2(nn.Module):
    """
    Actor-V2 model, using the Torch's official implementation of gat and pooling layers.
    """

    def __init__(self, n_mcs, agent=0, adj=None, n_heads=3, n_out_gat=100, L=5, ratio=0.8):
        """
        Args:
            n_out_gat: number of units of features extracted by GAT layers
            n_mcs: 微服务数量
            n_actions: 动作数
            agent: 服务器数量
            adj: 微服务的邻接矩阵
            n_heads: 多头注意力的头数
        """
        super(PoolActorV2, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(3 * n_mcs + 2, 100), nn.ELU())
        self.dense2 = nn.Sequential(nn.Linear(100 + n_heads * math.ceil(n_mcs * ratio) * n_out_gat + L * 2 * 150, 128),
                                    nn.ELU())
        self.dense_outs = nn.ModuleList([nn.Sequential(nn.Linear(128, 1), nn.LeakyReLU()) for _ in range(n_mcs)])
        # self.param_init()
        self.heads = n_heads
        self.agent = agent
        self.adj = adj
        self.gat2conv = GATv2Conv(in_channels=2 * agent + 2, out_channels=n_out_gat, heads=n_heads)
        self.pooling = TopKPooling(in_channels=n_heads * n_out_gat, ratio=ratio)

        # self.gat = GAT(2 * agent + 2, n_out=n_out_gat)
        self.bilstm = nn.LSTM(input_size=77 + n_mcs, hidden_size=150, bidirectional=True, num_layers=1,
                              batch_first=True)
        # self.add_module("gat", self.gat)
        self.n_mcs = n_mcs

        self.param_init()

    def param_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0, std=0.7)

    def forward(self, obs, past_obs):
        # obs: (B,F) past_obs: (B,L,F)
        B = obs.size(0) if isinstance(obs, torch.Tensor) else obs.shape(0)
        s = self.agent
        m = self.n_mcs
        edge_index, _ = dense_to_sparse(self.adj)

        # past_enc: (B,L,2*F)
        past_enc, (h, c) = self.bilstm(past_obs)
        past_enc = torch.reshape(past_enc, (past_enc.size()[0], -1))
        server_state = obs[:, :3 * m + 2]
        mcs_res = torch.reshape(obs[:, 3 * m + 2:4 * m + 2], (-1, m, 1))
        mcs_ins = torch.reshape(obs[:, 4 * m + 2:5 * m + 2], (-1, m, 1))
        mcs_resp_server = torch.reshape(obs[:, 5 * m + 2:5 * m + 2 + m * s], (-1, m, s))
        mcs_insp_server = torch.reshape(obs[:, 5 * m + 2 + m * s:], (-1, m, s))

        # 服务器图特征(B, m, f)
        mcs_feat = torch.concat([mcs_res, mcs_ins, mcs_resp_server, mcs_insp_server],
                                dim=-1)

        # (batch, m, n_heads * 80)
        batch_feats = []
        for x in mcs_feat:
            batch_feats.append(Data(x=x, edge_index=edge_index))
        batched_x = Batch.from_data_list(batch_feats)
        x = self.gat2conv(batched_x.x, batched_x.edge_index)
        x = torch.reshape(x, (B, m, -1))
        batch_feats.clear()
        for x_ in x:
            batch_feats.append(Data(x=x_, edge_index=edge_index))
        x = Batch.from_data_list(batch_feats)
        x = self.pooling(x.x, x.edge_index)
        x = torch.reshape(x[0], (B, -1))

        # for i in range(gat_feat.size(0)):
        #     pooled_gat_feat = self.pooling(gat_feat[i], edge_index=edge_index)
        #     batch_feats.append(pooled_gat_feat[0])
        # mcs_gat_feat = torch.stack(batch_feats)
        # # (B, (m/2)*F) 向上取整
        # mcs_gat_feat = torch.reshape(mcs_gat_feat, (mcs_gat_feat.size(0), -1))
        mcs_gat_feat = x

        # (batch, 500)
        server_feat = self.dense1(server_state)
        server_feat = torch.reshape(server_feat, (server_feat.size(0), -1))

        # (batch, 500 + m * n_heads * self.n_out_gat)
        hidden_feat = self.dense2(torch.cat([server_feat, mcs_gat_feat, past_enc], dim=-1))
        actions = []
        for i in range(self.n_mcs):
            actions.append(self.dense_outs[i](hidden_feat))
        # actions: (batch, n_mcs)
        actions = torch.cat(actions, dim=-1)
        return actions


class GALActor(nn.Module):
    def __init__(self, n_mcs, n_actions=0, agent=0, adj=None, n_heads=3,
                 n_hidden_gat=100, n_out_gat=100, L=5):
        """
        初始化函数
        Args:
            n_hidden_gat:
            n_out_gat:
            n_mcs: 微服务数量
            n_actions: 动作数
            agent: 服务器数量
            adj: 微服务的邻接矩阵
            n_heads: 多头注意力的头数
        """
        super(GALActor, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(3 * n_mcs + 2, 100), nn.ELU())
        self.dense2 = nn.Sequential(nn.Linear(100 + n_heads * n_mcs * n_out_gat + L * 2 * 150, 128),
                                    nn.ELU())
        self.dense_outs = nn.ModuleList([nn.Sequential(nn.Linear(128, n_actions),
                                                       nn.ELU()) for _ in range(n_mcs)])
        # self.param_init()
        self.agent = agent
        self.adj = adj
        self.gat = GAT(2 * agent + 2)
        self.bilstm = nn.LSTM(input_size=77 + 3 * n_mcs, hidden_size=150, bidirectional=True, num_layers=1,
                              batch_first=True)
        self.add_module("gat", self.gat)
        self.n_mcs = n_mcs

        self.param_init()

    def param_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.normal_(module.bias, mean=0, std=0.7)

    def forward(self, obs, past_obs):
        # obs: (B,F) past_obs: (B,L,F)
        s = self.agent
        m = self.n_mcs

        # past_enc: (B,L,2*F)
        past_enc, (h, c) = self.bilstm(past_obs)
        past_enc = torch.reshape(past_enc, (past_enc.size()[0], -1))
        server_state = obs[:, :3 * m + 2]
        mcs_res = torch.reshape(obs[:, 3 * m + 2:4 * m + 2], (-1, m, 1))
        mcs_ins = torch.reshape(obs[:, 4 * m + 2:5 * m + 2], (-1, m, 1))
        mcs_resp_server = torch.reshape(obs[:, 5 * m + 2:5 * m + 2 + m * s], (-1, m, s))
        mcs_insp_server = torch.reshape(obs[:, 5 * m + 2 + m * s:], (-1, m, s))

        # 服务器图特征(m, f)
        mcs_feat = torch.concat([mcs_res, mcs_ins, mcs_resp_server, mcs_insp_server],
                                dim=-1)
        # (batch, m, n_heads * 80)
        mcs_gat_feat = self.gat(mcs_feat, self.adj)
        mcs_gat_feat = torch.reshape(mcs_gat_feat, (mcs_gat_feat.size()[0], -1))
        # (batch, 500)
        server_feat = F.relu(self.dense1(server_state))
        server_feat = torch.reshape(server_feat, (server_feat.size()[0], -1))

        # (batch, 500 + m * n_heads * self.n_out_gat)
        hidden_feat = F.relu(self.dense2(torch.cat([server_feat, mcs_gat_feat, past_enc], dim=-1)))
        actions = []
        for i in range(self.n_mcs):
            actions.append(F.gumbel_softmax(F.tanh(self.dense_outs[i](hidden_feat)),
                                            tau=1, dim=-1, hard=True))
        # actions: (batch, n_mcs x n_acitons)
        actions = torch.cat(actions, dim=-1)
        return actions


class GAT(nn.Module):
    def __init__(self, nfeat, n_out=100, dropout=0.4, alpha=0.01, nheads=3):
        """Dense version of GAT.

        Args:
            n_out:
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([GraphAttentionLayer(in_features=nfeat,
                                                             out_features=n_out, n_heads=nheads, dropout=dropout,
                                                             alpha=alpha,
                                                             concat=True)])

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu_(torch.cat([att(x, adj) for att in self.attentions], dim=-1))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, n_heads=3,
                 dropout=0.5, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha
        self.concat = concat

        self.W = nn.ParameterList([nn.Parameter(torch.empty(size=(in_features, out_features)))
                                   for _ in range(n_heads)])

        self.a = nn.ParameterList([nn.Parameter(torch.empty(size=(2 * out_features, 1)))
                                   for _ in range(n_heads)])
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layernorm = nn.LayerNorm(n_heads * out_features)

        self.param_init()

    def param_init(self):
        for w, a_ in zip(self.W, self.a):
            nn.init.orthogonal_(w)
            nn.init.orthogonal_(a_)
        nn.init.normal_(self.layernorm.weight, mean=0, std=0.5)

    def forward(self, h, adj):
        feats = []
        for head in range(self.n_heads):
            # h(N,in_F) W(in_F,out_F) Wh(N,out_F)
            Wh = torch.matmul(h, self.W[head])
            attention = self.attention_score(Wh, adj, self.a[head])
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)
            feats.append(F.elu_(h_prime))
        return self.layernorm(torch.cat(feats, dim=-1))

    def attention_score(self, h, adj, att_mx):
        # h.shape (N, out_feature)
        # attention (N, N)
        N = h.size()[-2]
        f = h.size()[-1]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, f), h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * f)
        e = F.leaky_relu(torch.squeeze(torch.matmul(a_input, att_mx)))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Masked attention scores
        attention = F.softmax(attention, dim=1)
        return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# [reference] https://gitorchub.com/mattorchiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://matorch.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, torcheta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min,
                                                       n_steps_annealing=n_steps_annealing)
        self.torcheta = torcheta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.torcheta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(loc=0, scale=10, size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def set_size(self, size: [tuple, list]):
        self.size = size


def x(*a):
    r = []
    for i in a:
        r.append(i)
    return r


if __name__ == '__main__':
    model = GAT(1, dropout=4, alpha=5, nheads=1)
    actor = Actor(30).to(device)
    a = torch.nn.Parameter(torch.rand(1, 2), requires_grad=True)
    for name, module in actor.named_modules():
        print("模块：", name, module)
    for name, param in actor.named_parameters():
        print(name, param.size())
    print("=====")
    print(x(*zip(*list(actor.gat.named_parameters())))[0])
    # for param, v in actor.state_dict().items():
    #     print(param)
    print(torch.equal(list(actor.named_parameters())[0][1], list(actor.parameters())[0]))
    print(isinstance(list(actor.named_parameters())[0][1], torch.nn.Linear))
    # for name, param in actor.named_parameters():
    #     print(name, param.size())
