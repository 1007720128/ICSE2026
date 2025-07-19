import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(__file__))
[sys.path.append(str(i)) for i in Path(os.path.dirname(__file__)).parents]

from standarize_stream import RunningMeanStd

import torch
import torch.nn as nn
from torch.optim import Adam
from memory import ReplayMemory, Experience
from memoryLstm import ReplayMemoryLstm, ExperienceLstm
from model import Critic, Actor, OrnsteinUhlenbeckProcess, GALActor, PoolActorV2, PoolCritic
from params import scale_reward
from torch.optim.lr_scheduler import StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MaddpgGat(nn.Module):
    def __init__(self, n_agents, dim_obs, n_mcs, batch_size=120, capacity=1000000, episodes_before_train=100,
                 n_actions=0, adj=None, mode="train", actor_model=None, critic_model=None):
        super().__init__()
        if not actor_model:
            actor_model = Actor
        if not critic_model:
            critic_model = Critic
        self.actors = nn.ModuleList([actor_model(n_mcs, n_actions=n_actions, agent=n_agents, adj=adj)
                                     for _ in range(n_agents)])
        self.critics = nn.ModuleList([critic_model(n_agents, dim_obs, n_actions=n_actions)
                                      for _ in range(n_agents)])
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = n_actions
        self.n_mcs = n_mcs
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.gradient_clip = 3
        self.GAMMA = 0.95
        self.tau = 0.01
        self.reward_norm = RunningMeanStd(shape=(self.n_agents,), device=device)

        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.003, eps=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.001, eps=0.001) for x in self.actors]
        self.actor_scheduler = [StepLR(optim, 450, gamma=0.8) for optim in self.actor_optimizer]

        print("当前设备：{}".format(device))
        for x in self.actors:
            x.to(device)
        for x in self.critics:
            x.to(device)
        for x in self.actors_target:
            x.to(device)
        for x in self.critics_target:
            x.to(device)

        self.steps_done = 0
        self.episode_done = 0
        self.random_pr = OrnsteinUhlenbeckProcess(2)
        self.set_mode(mode)

    def set_mode(self, mode: str = "train"):
        if mode == "train":
            self.train()
        else:
            self.eval()

    def alter_adj(self, adj):
        print(len(self.actors))
        for actor in self.actors:
            actor.adj = adj

    def update_policy(self):
        # 目标函数其实就是最大化Q函数的期望
        # do not train until exploration is enough
        # if self.episode_done <= self.episodes_before_train:
        #     return None, None
        if len(self.memory) < self.batch_size * 50 or self.steps_done < self.batch_size:
            print("当前步数不足，当前步数为：{}，期望最低步数为：{}，不进行训练".format(
                self.steps_done, self.batch_size))
            return None, None

        print("=======开始训练!=========")

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            # 布尔索引，mask None状态
            non_final_mask = list(map(lambda s: s is not None, batch.next_states))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).float()
            action_batch = torch.stack(batch.actions).float()
            reward_batch = torch.stack(batch.rewards).float()
            # (batch,agent,obs)
            non_final_next_states = torch.stack([s for s in batch.next_states
                                                 if s is not None]).float()

            q_loss, whole_state = self.update_critic(action_batch, agent, non_final_mask, non_final_next_states,
                                                     reward_batch, state_batch)

            actor_loss = self.update_actor(action_batch, agent, state_batch, whole_state)
            c_loss.append(q_loss)
            a_loss.append(actor_loss)

        # 周期性的复制逼近网络的参数到目标网络
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        print("========训练结束!==========")
        return c_loss, a_loss

    def update_actor(self, action_batch, agent, state_batch, whole_state):
        # 更新actor的参数
        self.actor_optimizer[agent].zero_grad()
        state_i = state_batch[:, agent, :]
        action_i = self.actors[agent](state_i)
        ac = action_batch.clone()
        ac[:, agent, :] = action_i
        whole_action = ac.view(self.batch_size, -1)
        # 这里因为critic网络是逼近的（s,a)价值函数，所以打得分越高，loss直觉上
        # 应该是越低的，因此这里简单的加了个负号，最大化Q函数的值
        actor_loss = -self.critics[agent](whole_state, whole_action)
        params = list(map(torch.clone, self.critics[agent].parameters()))
        param_actors = deepcopy(dict(self.actors[agent].named_parameters()))
        # 对一个batch的loss求平均
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.gradient_clip)
        self.actor_optimizer[agent].step()
        params1 = list(map(torch.clone, self.critics[agent].parameters()))
        param_actors_postfix = dict(self.actors[agent].named_parameters())
        # for name, param in param_actors_postfix.items():
        #     if len(param.grad[param.grad > 0]) > 0:
        #         print("更新到的参数:{}".format(name))
        return actor_loss

    def update_critic(self, action_batch, agent, non_final_mask, non_final_next_states, reward_batch, state_batch):
        # update q function for current agent
        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)
        self.critic_optimizer[agent].zero_grad()
        # 这个地方每个agent都是学的(s,a),s代表当前状态,a则是基于当前状态作出的动作
        q = self.critics[agent](whole_state, whole_action)
        # end状态的样本，对应的y_hat为0
        non_final_next_actions = [
            self.actors_target[i](non_final_next_states[:,
                                  i,
                                  :]) for i in range(self.n_agents)]
        non_final_next_actions = torch.stack(non_final_next_actions)
        # 此处.contiguous()是为了保证view方法不报错，参考链接：https://zhuanlan.zhihu.com/p/64551412
        non_final_next_actions = (
            non_final_next_actions.transpose(0,
                                             1).contiguous())
        y_hat = torch.zeros(self.batch_size).type(torch.float32).to(device)
        y_hat[non_final_mask] = self.critics_target[agent](
            non_final_next_states.view((-1, self.n_agents * self.n_states)),
            non_final_next_actions.view((-1, self.n_agents * self.n_mcs * self.n_actions))).squeeze()
        # scale_reward: to scale reward in Q functions
        # 目标网络的Q值考虑了奖励函数
        # self.reward_norm.update(reward_batch)
        # reward_batch = (reward_batch - self.reward_norm.mean) / torch.sqrt(self.reward_norm.var)
        y_hat = (y_hat.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)
        q_loss = nn.SmoothL1Loss()(q, y_hat.detach())
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.gradient_clip)
        self.critic_optimizer[agent].step()
        return q_loss, whole_state

    def select_action(self, state_batch):
        """
        根据环境状态为每个智能体选择动作，state_batch的形状是[n_agents , state_dim]
        n_agents表示智能体的个数，state_dim是状态的维度

        :param state_batch:
        :return:
        """
        # state_batch: (batch, n_agents, state_dim)
        actions = torch.zeros(self.n_agents, self.n_mcs * self.n_actions)
        for i in range(self.n_agents):
            # sb: (batch, state_dim)
            sb = state_batch[:, i, :].detach()
            act = self.actors[i](sb)
            actions[i, ...] = act
        return actions

    def save_model(self, model_dir):
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), Path(model_dir, "actor {}".format(i)))
            torch.save(self.critics[i].state_dict(), Path(model_dir, "critic {}".format(i)))

    def load_model(self, model_dir):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(
                Path(model_dir, "actor {}".format(i)), map_location=device))
            self.critics[i].load_state_dict(torch.load(
                Path(model_dir, "critic {}".format(i)), map_location=device))


class MaddpgLstm(nn.Module):
    def __init__(self, n_agents, dim_obs, n_mcs, batch_size=120, capacity=1000000, episodes_before_train=100,
                 n_actions=0, adj=None, mode="train", actor_model=None, critic_model=None, L=5):
        super().__init__()
        if not actor_model:
            actor_model = GALActor
        if not critic_model:
            critic_model = Critic
        self.actors = nn.ModuleList([actor_model(n_mcs, n_actions=n_actions, agent=n_agents, adj=adj)
                                     for _ in range(n_agents)])
        self.critics = nn.ModuleList([critic_model(n_agents, dim_obs, n_actions=n_actions)
                                      for _ in range(n_agents)])
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = n_actions
        self.n_mcs = n_mcs
        self.memory = ReplayMemoryLstm(capacity)
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.gradient_clip = 3
        self.GAMMA = 0.95
        self.tau = 0.01
        self.reward_norm = RunningMeanStd(shape=(self.n_agents,), device=device)

        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.003, eps=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.001, eps=0.001) for x in self.actors]
        self.actor_scheduler = [StepLR(optim, 450, gamma=0.8) for optim in self.actor_optimizer]

        self.L = L
        self.past_inf = []
        print("当前设备：{}".format(device))
        for x in self.actors:
            x.to(device)
        for x in self.critics:
            x.to(device)
        for x in self.actors_target:
            x.to(device)
        for x in self.critics_target:
            x.to(device)

        self.steps_done = 0
        self.episode_done = 0
        self.random_pr = OrnsteinUhlenbeckProcess(2)
        self.set_mode(mode)

    def get_pastinf(self):
        padding_zeros = torch.zeros(self.L - len(self.past_inf), self.n_agents, 92)
        return torch.cat([padding_zeros, torch.stack(self.past_inf) if self.past_inf else torch.tensor([])], dim=0)

    def push_past_inf(self, curinf: torch.Tensor):
        self.past_inf = self.past_inf[-(self.L - 1):] + [curinf]

    def set_mode(self, mode: str = "train"):
        if mode == "train":
            self.train()
        else:
            self.eval()

    def patch_adj(self, adj):
        print(len(self.actors))
        for actor in self.actors:
            actor.adj = adj
            actor.n_mcs = len(adj)

    def update_policy(self):
        # 目标函数其实就是最大化Q函数的期望
        # do not train until exploration is enough
        # if self.episode_done <= self.episodes_before_train:
        #     return None, None
        if len(self.memory) < self.batch_size * 5 or self.steps_done < self.batch_size:
            print("当前步数不足，当前步数为：{}，期望最低步数为：{}，不进行训练".format(
                self.steps_done, self.batch_size))
            return None, None

        print("=======开始训练!=========")

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = ExperienceLstm(*zip(*transitions))
            # 布尔索引，mask None状态
            non_final_mask = list(map(lambda s: s is not None, batch.next_states))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).float()
            action_batch = torch.stack(batch.actions).float()
            reward_batch = torch.stack(batch.rewards).float()
            pastinf_batch = torch.stack(batch.past_inf).float()
            # (batch,agent,obs)
            non_final_next_states = torch.stack([s for s in batch.next_states
                                                 if s is not None]).float()

            q_loss, whole_state = self.update_critic(action_batch, agent, non_final_mask, non_final_next_states,
                                                     reward_batch, state_batch, pastinf_batch)

            actor_loss = self.update_actor(action_batch, agent, state_batch, whole_state, pastinf_batch)
            c_loss.append(q_loss)
            a_loss.append(actor_loss)

        # 周期性的复制逼近网络的参数到目标网络
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        print("========训练结束!==========")
        return c_loss, a_loss

    def update_actor(self, action_batch, agent, state_batch, whole_state, pastinf_batch):
        # 更新actor的参数
        self.actor_optimizer[agent].zero_grad()
        state_i = state_batch[:, agent, :]
        action_i = self.actors[agent](state_i, pastinf_batch[:, :, agent, :])
        ac = action_batch.clone()
        ac[:, agent, :] = action_i
        whole_action = ac.view(self.batch_size, -1)
        # 这里因为critic网络是逼近的（s,a)价值函数，所以打得分越高，loss直觉上
        # 应该是越低的，因此这里简单的加了个负号，最大化Q函数的值
        actor_loss = -self.critics[agent](whole_state, whole_action)
        params = list(map(torch.clone, self.critics[agent].parameters()))
        param_actors = deepcopy(dict(self.actors[agent].named_parameters()))
        # 对一个batch的loss求平均
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.gradient_clip)
        self.actor_optimizer[agent].step()
        params1 = list(map(torch.clone, self.critics[agent].parameters()))
        param_actors_postfix = dict(self.actors[agent].named_parameters())
        # for name, param in param_actors_postfix.items():
        #     if len(param.grad[param.grad > 0]) > 0:
        #         print("更新到的参数:{}".format(name))
        return actor_loss

    def update_critic(self, action_batch, agent, non_final_mask, non_final_next_states, reward_batch, state_batch,
                      pastinf_batch):
        # update q function for current agent
        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)
        self.critic_optimizer[agent].zero_grad()
        # 这个地方每个agent都是学的(s,a),s代表当前状态,a则是基于当前状态作出的动作
        q = self.critics[agent](whole_state, whole_action)
        # end状态的样本，对应的y_hat为0
        non_final_next_actions = [
            self.actors_target[i](non_final_next_states[:,
                                  i,
                                  :], pastinf_batch[non_final_mask, :, i, :]) for i in range(self.n_agents)]
        non_final_next_actions = torch.stack(non_final_next_actions)
        # 此处.contiguous()是为了保证view方法不报错，参考链接：https://zhuanlan.zhihu.com/p/64551412
        non_final_next_actions = (
            non_final_next_actions.transpose(0,
                                             1).contiguous())
        y_hat = torch.zeros(self.batch_size).type(torch.float32).to(device)
        y_hat[non_final_mask] = self.critics_target[agent](
            non_final_next_states.view((-1, self.n_agents * self.n_states)),
            non_final_next_actions.view((-1, self.n_agents * self.n_mcs * self.n_actions))).squeeze()
        # scale_reward: to scale reward in Q functions
        # 目标网络的Q值考虑了奖励函数
        # self.reward_norm.update(reward_batch)
        # reward_batch = (reward_batch - self.reward_norm.mean) / torch.sqrt(self.reward_norm.var)
        y_hat = (y_hat.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)
        q_loss = nn.SmoothL1Loss()(q, y_hat.detach())
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.gradient_clip)
        self.critic_optimizer[agent].step()
        return q_loss, whole_state

    def select_action(self, state_batch):
        """
        根据环境状态为每个智能体选择动作，state_batch的形状是[n_agents , state_dim]
        n_agents表示智能体的个数，state_dim是状态的维度

        :param state_batch:
        :return:
        """
        # state_batch: (batch, n_agents, state_dim)
        actions = torch.zeros(self.n_agents, self.n_mcs * self.n_actions)
        for i in range(self.n_agents):
            # sb: (batch, state_dim) pastinf: (B, L, F)
            pastinf = torch.unsqueeze(self.get_pastinf()[:, i], dim=0)
            sb = state_batch[:, i, :].detach()

            act = self.actors[i](sb, pastinf)
            actions[i, ...] = act
        return actions

    def save_model(self, model_dir):
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), Path(model_dir, "actor {}".format(i)))
            torch.save(self.critics[i].state_dict(), Path(model_dir, "critic {}".format(i)))

    def load_model(self, model_dir):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(
                Path(model_dir, "actor {}".format(i)), map_location=device))
            self.critics[i].load_state_dict(torch.load(
                Path(model_dir, "critic {}".format(i)), map_location=device))


class MaddpgPool(nn.Module):
    def __init__(self, n_agents, dim_obs, n_mcs, batch_size=120, capacity=1000000, episodes_before_train=100,
                 n_actions=0, adj=None, mode="train", actor_model=None, critic_model=None, L=5):
        super().__init__()
        if not actor_model:
            actor_model = PoolActorV2
        if not critic_model:
            critic_model = PoolCritic
        self.actors = nn.ModuleList([actor_model(n_mcs=n_mcs, agent=n_agents, adj=adj)
                                     for _ in range(n_agents)])
        self.critics = nn.ModuleList([critic_model(n_agents, dim_obs, n_actions=n_actions)
                                      for _ in range(n_agents)])
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = n_actions
        self.n_mcs = n_mcs
        self.memory = ReplayMemoryLstm(capacity)
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.gradient_clip = 3
        self.GAMMA = 0.95
        self.tau = 0.01
        self.reward_norm = RunningMeanStd(shape=(self.n_agents,), device=device)

        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.003, eps=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.001, eps=0.001) for x in self.actors]
        self.actor_scheduler = [StepLR(optim, 450, gamma=0.8) for optim in self.actor_optimizer]

        self.L = L
        self.past_inf = []
        print("当前设备：{}".format(device))
        for x in self.actors:
            x.to(device)
        for x in self.critics:
            x.to(device)
        for x in self.actors_target:
            x.to(device)
        for x in self.critics_target:
            x.to(device)

        self.steps_done = 0
        self.episode_done = 0
        self.random_pr = OrnsteinUhlenbeckProcess(2)
        self.set_mode(mode)

    def get_pastinf(self):
        padding_zeros = torch.zeros(self.L - len(self.past_inf), self.n_agents, 82)
        return torch.cat([padding_zeros, torch.stack(self.past_inf) if self.past_inf else torch.tensor([])], dim=0)

    def push_past_inf(self, curinf: torch.Tensor):
        self.past_inf = self.past_inf[-(self.L - 1):] + [curinf]

    def set_mode(self, mode: str = "train"):
        if mode == "train":
            self.train()
        else:
            self.eval()

    def alter_adj(self, adj):
        print(len(self.actors))
        for actor in self.actors:
            actor.adj = adj

    def update_policy(self):
        # 目标函数其实就是最大化Q函数的期望
        # do not train until exploration is enough
        # if self.episode_done <= self.episodes_before_train:
        #     return None, None
        if len(self.memory) < self.batch_size * 5 or self.steps_done < self.batch_size:
            print("当前步数不足，当前步数为：{}，期望最低步数为：{}，不进行训练".format(
                self.steps_done, self.batch_size))
            return None, None

        print("=======开始训练!=========")

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = ExperienceLstm(*zip(*transitions))
            # 布尔索引，mask None状态
            non_final_mask = list(map(lambda s: s is not None, batch.next_states))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).float()
            action_batch = torch.stack(batch.actions).float()
            reward_batch = torch.stack(batch.rewards).float()
            pastinf_batch = torch.stack(batch.past_inf).float()
            # (batch,agent,obs)
            non_final_next_states = torch.stack([s for s in batch.next_states
                                                 if s is not None]).float()

            q_loss, whole_state = self.update_critic(action_batch, agent, non_final_mask, non_final_next_states,
                                                     reward_batch, state_batch, pastinf_batch)

            actor_loss = self.update_actor(action_batch, agent, state_batch, whole_state, pastinf_batch)
            c_loss.append(q_loss)
            a_loss.append(actor_loss)

        # 周期性的复制逼近网络的参数到目标网络
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        print("========训练结束!==========")
        return c_loss, a_loss

    def update_actor(self, action_batch, agent, state_batch, whole_state, pastinf_batch):
        # 更新actor的参数
        self.actor_optimizer[agent].zero_grad()
        state_i = state_batch[:, agent, :]
        action_i = self.actors[agent](state_i, pastinf_batch[:, :, agent, :])
        ac = action_batch.clone()
        ac[:, agent, :] = action_i
        whole_action = ac.view(self.batch_size, -1)
        # 这里因为critic网络是逼近的（s,a)价值函数，所以打得分越高，loss直觉上
        # 应该是越低的，因此这里简单的加了个负号，最大化Q函数的值
        actor_loss = -self.critics[agent](whole_state, whole_action)

        # 对一个batch的loss求平均
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        all_params = list(zip(*self.actors[agent].named_parameters()))[0]
        unchanged_params = []
        for name, param in self.actors[agent].named_parameters():
            if param.grad[param.grad != 0].numel() == 0:
                unchanged_params.append(name)
        if len(unchanged_params) == len(all_params):
            print("全部未更新")
        else:
            print("未更新参数：", unchanged_params)
            print("更新参数：", [i for i in all_params if i not in unchanged_params])

        # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.gradient_clip)
        self.actor_optimizer[agent].step()

        params1 = list(map(torch.clone, self.critics[agent].parameters()))
        param_actors_postfix = dict(self.actors[agent].named_parameters())
        # for name, param in param_actors_postfix.items():
        #     print(name, param.grad)
        #     if len(param.grad[param.grad > 0]) > 0:
        #         print("更新到的参数:{}".format(name))
        return actor_loss

    def update_critic(self, action_batch, agent, non_final_mask, non_final_next_states, reward_batch, state_batch,
                      pastinf_batch):
        # update q function for current agent
        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)
        self.critic_optimizer[agent].zero_grad()
        # 这个地方每个agent都是学的(s,a),s代表当前状态,a则是基于当前状态作出的动作
        q = self.critics[agent](whole_state, whole_action)
        # end状态的样本，对应的y_hat为0
        non_final_next_actions = [
            self.actors_target[i](non_final_next_states[:,
                                  i,
                                  :], pastinf_batch[non_final_mask, :, i, :]) for i in range(self.n_agents)]
        non_final_next_actions = torch.stack(non_final_next_actions)
        # 此处.contiguous()是为了保证view方法不报错，参考链接：https://zhuanlan.zhihu.com/p/64551412
        non_final_next_actions = (
            non_final_next_actions.transpose(0,
                                             1).contiguous())
        y_hat = torch.zeros(self.batch_size).type(torch.float32).to(device)
        y_hat[non_final_mask] = self.critics_target[agent](
            non_final_next_states.view((non_final_mask.count(True), -1)),
            non_final_next_actions.view((non_final_mask.count(True), -1))).squeeze()
        # scale_reward: to scale reward in Q functions
        # 目标网络的Q值考虑了奖励函数
        # self.reward_norm.update(reward_batch)
        # reward_batch = (reward_batch - self.reward_norm.mean) / torch.sqrt(self.reward_norm.var)
        y_hat = (y_hat.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)
        q_loss = nn.SmoothL1Loss()(q, y_hat.detach())
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.gradient_clip)
        self.critic_optimizer[agent].step()
        return q_loss, whole_state

    def select_action(self, state_batch):
        """
        根据环境状态为每个智能体选择动作，state_batch的形状是[n_agents , state_dim]
        n_agents表示智能体的个数，state_dim是状态的维度

        :param state_batch:
        :return:
        """
        # state_batch: (batch, n_agents, state_dim)
        actions = torch.zeros(self.n_agents, self.n_mcs)
        for i in range(self.n_agents):
            # sb: (batch, state_dim) pastinf: (B, L, F)
            pastinf = torch.unsqueeze(self.get_pastinf()[:, i], dim=0)
            sb = state_batch[:, i, :].detach()

            act = self.actors[i](sb, pastinf) + torch.randint(-5, 5, (self.n_mcs,))
            actions[i, ...] = act
        return actions

    def save_model(self, model_dir):
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), Path(model_dir, "actor {}".format(i)))
            torch.save(self.critics[i].state_dict(), Path(model_dir, "critic {}".format(i)))

    def load_model(self, model_dir):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(
                Path(model_dir, "actor {}".format(i)), map_location=device))
            self.critics[i].load_state_dict(torch.load(
                Path(model_dir, "critic {}".format(i)), map_location=device))
