import math
import os
import os.path
import pickle
import shutil
import time
import warnings
from functools import wraps
from multiprocessing import Process
from pathlib import Path
from typing import Tuple, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from mpl_toolkits import axisartist
from numpy import ndarray as arr
from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, SMAPE
from scipy.stats import norm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from algos.AdaptiveScaler import GR_MAPPO, GRMAPPOPolicy
from algos.gnn_util import format_training_duration, compute_mcs_edge_adj, \
    compute_disseminated_workload, calculate_sla_violation_rate, calculate_p95_latency
from algos.maddpg_torch import MaddpgGat, MaddpgLstm, MaddpgPool
# from simulation.utils.metrics import calculate_sla_violation_rate
from algos.ppo_buffer import GraphReplayBuffer

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_dir = Path(os.path.dirname(__file__))
models_dir = Path(cur_dir, "模型状态保存")
models_dir.mkdir(exist_ok=True)
gma_model = Path(models_dir, "gma")
gma_model.mkdir(exist_ok=True)
results_dir = Path("/Users/username/Project/GatMicroservice/simulation/Evaluation", "Results")
results_dir.mkdir(exist_ok=True, parents=True)
pict_dir = Path("/Users/username/Project/GatMicroservice/simulation/Evaluation", "Pictures")
pict_dir.mkdir(exist_ok=True, parents=True)

MS = int(1e3)

KB = int(8 * 1e3)
MB = int(8 * 1e6)
GB = int(8 * 1e9)

MHZ = int(1e6)
GHZ = int(1e9)

tick_size = 20
label_size = 20
legend_size = 15
title_size = 20
text_size = 15


# 用于监控单个进程内存的函数
def measure_memory(pid):
    process = psutil.Process(pid)
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
    return memory_usage


def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        start_cpu_times = psutil.cpu_times()
        start_mem = process.memory_info().rss
        start_cpu_percent = psutil.cpu_percent(percpu=True)
        start_cpu = psutil.cpu_percent()
        start_time = time.time()
        result = func(*args, **kwargs)

        # process = Process(target=func)
        # process.start()
        #
        # # 获取子进程的PID
        # pid = process.pid
        #
        # # 记录推理前的内存使用情况
        # memory_before = measure_memory(pid)
        #
        # # 等待子进程执行完成
        # process.join()
        #
        # # 记录推理后的内存使用情况
        # memory_after = measure_memory(pid)
        #
        # # 计算内存使用变化
        # memory_used = memory_after - memory_before

        end_time = time.time()
        end_cpu_times = psutil.cpu_times()
        end_mem = process.memory_info().rss
        end_cpu_percent = psutil.cpu_percent(percpu=True)
        end_cpu = psutil.cpu_percent()

        # 计算CPU时间（以秒为单位）
        cpu_time = sum(end_cpu_times) - sum(start_cpu_times)

        print(f"CPU时间: {cpu_time:.3f} 核秒")
        print(f"CPU使用率: {end_cpu - start_cpu}%")
        print(f"内存使用: {(end_mem - start_mem) / (1024 * 1024):.2f} MB")
        print(f"运行时间: {end_time - start_time:.2f} 秒")

        return end_time - start_time, (end_mem - start_mem) / (1024 * 1024)

    return wrapper


class Metrics:
    """
    此类保存根据香农公式、排队论计算的指标值，例如响应时间的计算
    """
    # 在空气中，电磁波速度约等于30万公里/s
    SpeedOfElectro = 3 * int(1e5)

    LatencyOfCloud = 50

    @staticmethod
    def shannon_formula_db(B, SN):
        """
           传输时延：根据香农公式计算信道容量，信噪比以db形式输入
           Args:
               B (float): 带宽/HZ
               SN (float): 信噪比/db

           Returns:
               信道容量 : 单位/mbps
       """
        # db转换，参考信噪比转换公式
        _SN = np.log10(SN / 10)
        return B * np.log2(1 + _SN)

    @staticmethod
    def shannon_formula(B, S, N):
        """
            传输时延：根据香农公式计算信道容量
            Args:
                B (float): 带宽/HZ
                S (float): 信号频率/W
                N (float): 噪声频率/W

            Returns:
                信道容量 : 单位/mbps
        """
        return B * np.log2(1 + S / N)

    @staticmethod
    def send_delay(data_size, channel_capacity):
        """
        发送时延，数据包推至链路需要的时间
        Args:
            data_size: 数据大小/mb
            channel_capacity: 信道容量/mbps

        Returns:
            发送时延/ms
        """
        return round((data_size / channel_capacity) * MS)

    @staticmethod
    def process_time(u_lambda, mu=100, instance=1):
        """
            处理时延：根据排队论的 M/M/1 模型，计算每个请求的平均处理时间，使用round-robin policy选择处理实例的请求
            :param mu: 离开速率
            :param u_lambda: 泊松分布参数，到达速率
            :param instance：服务已部署的实例数
            :return: 处理时间/ms
        """
        # 资源竞争扰动因子
        omega = torch.nn.init.trunc_normal_(torch.zeros((1,)), a=1, b=1.08).item()
        # omega = torch.nn.init.trunc_normal_(torch.zeros((1,)), a=1, b=1.5).item()
        # init_omega = np.random.uniform(2, 3)
        prc_time = 0
        is_init = False
        threshold_coefficient = int(0.93 * mu)
        # if instance == 1:
        #     is_init = True
        #     instance = 2
        # 过载，用户到达速率逼近极限，进行排队分流
        while math.ceil(u_lambda / instance) >= threshold_coefficient:
            prc_time += round(omega * (1 / (mu - threshold_coefficient)) * MS)
            u_lambda -= threshold_coefficient
        prc_time += round(omega * (1 / (mu - math.ceil(u_lambda / instance))) * MS, 3)
        return round(prc_time, 3)

    @staticmethod
    def propagation_time(distance):
        """
        信号在介质中的传输时延/ms，此时延只与距离有关
        Args:
            distance: 传输距离/km

        Returns:
            传输时延/ms
        """
        return round((distance / Metrics.SpeedOfElectro) * MS)


class Application:
    # 默认保存生成应用程序的路径
    cur_dir = Path(os.path.dirname(__file__))
    sv_dir = Path(cur_dir, "微服务结构")
    sv_dir.mkdir(parents=True, exist_ok=True)
    graph_file = Path(cur_dir, "VPA服务应用图.pkl")

    def __init__(self, is_gen=False, adj_construct=None):
        self.n_microservice = None
        self.G = nx.DiGraph()
        self.microservices = []
        self.adj = None
        self.cycles = []
        if not is_gen:
            try:
                self.load_graph(file_pth="/Users/username/Project/Paper-Scalable/simulation/微服务结构/graph_9.pkl")
            except FileNotFoundError as e:
                print(e, "图文件不存在，重新初始化图")
                self.gen_graph()
            except EOFError:
                print("文件格式错误，重新初始化图")
                self.gen_graph()
        else:
            self.gen_graph(adj=adj_construct)

    def get_upstream_services(self, svc: int):
        return nx.ancestors(self.G, svc)

    def gen_graph(self, adj=None, n_microservice=8, is_draw=False):
        self.prepare_files()
        self.G.clear()
        self.microservices.clear()
        if adj is None:
            self.n_microservice = n_microservice
            self.adj = torch.randint(0, 2, (self.n_microservice, self.n_microservice)).numpy()
        else:
            self.n_microservice = len(adj)
            self.adj = adj

        # 构建依赖图
        for i in range(self.n_microservice):
            self.microservices.append(Microservice(i))
        for i in range(self.n_microservice):
            self.G.add_node(i, color="red")
            for j in range(self.n_microservice):
                if i == j:
                    continue
                if self.adj[i, j] == 1:
                    self.G.add_edge(i, j, weight=3, comment="ok")
                    self.microservices[i].children.append(j)

        # 去除图中的环，使其成为DAG图
        self.normalize_to_dag()
        if is_draw:
            self.draw_graph()

    @staticmethod
    def prepare_files():
        """
            Prepare and organize graph files by ensuring the current directory has no graph files,
            and that the files in the historical graph folder (sv_dir) are properly ordered.

            This function performs the following tasks:
            1. Retrieves all graph files in the current directory and sorts them by their numbering.
            2. Retrieves and sorts all graph files in the historical directory (sv_dir).
            3. Removes old graph files in sv_dir, keeping only the latest `mx_files` number of files.
            4. Renames and moves the graph files from the current directory to the historical folder,
               ensuring proper ordering.
        """
        files = []
        mx_files = 20
        for f in Application.cur_dir.iterdir():
            if Path.is_file(f) and f.name.startswith("graph"):
                files.append(f)
        files.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
        sv_files = []
        for f in Application.sv_dir.iterdir():
            if Path.is_file(f) and f.name.startswith("graph"):
                sv_files.append(f)
        sv_files.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
        for f in sv_files[:-1 * mx_files]:
            if f.is_file():
                print(f"Removing old file: {f}")
                f.unlink()
        sv_files = sv_files[-1 * mx_files:]
        for i in range(len(sv_files)):
            shutil.move(Path(Application.sv_dir, sv_files[i]),
                        Path(Application.sv_dir, "graph_{}.pkl".format(i)))
        for i in range(len(files)):
            shutil.move(Path(Application.cur_dir, files[i]),
                        Path(Application.sv_dir, "graph_{}.pkl".format(len(sv_files) + i)))

    def normalize_to_dag(self):
        """
            Convert the graph to a Directed Acyclic Graph (DAG).

            This method removes cycles from the graph by iteratively detecting and
            removing edges that form cycles. It ensures that the graph becomes
            acyclic, following the properties of a DAG.

            The function works by repeatedly identifying cycles and removing edges
            until no cycles remain in the graph. The graph's structure is adjusted
            to ensure it remains directed and acyclic.
        """
        self.find_cycles()
        print("Current cycles: {}".format(self.cycles))
        while len(self.cycles) >= 1:
            for cycle in self.cycles:
                edge0, edge1 = cycle[-2], cycle[-1]
                print("Attempting to remove edge {}-{}".format(edge0, edge1))
                try:
                    self.adj[edge0, edge1] = 0
                    self.microservices[edge0].children.remove(edge1)
                    self.G.remove_edge(edge0, edge1)
                except (nx.exception.NetworkXError, ValueError):
                    print("Edge {}-{} already removed".format(edge0, edge1))
                break
            self.cycles = []
            self.find_cycles()
            print("Current cycles after removal: {}".format(self.cycles))
        self.save_graph()

    def find_cycles(self):
        """
           Find all cycles in the graph.
           This method performs a depth-first search (DFS) to identify all cycles
           in the graph represented by the adjacency matrix `self.adj`.
        """

        def _dfs(path):
            """
            Perform depth-first search to find cycles in the graph.
            Args:
                path (list): The current traversal path, starting from a node.
            """
            for m in np.where(self.adj[path[-1]])[0]:
                if m in path:
                    if path[path.index(m):] + [m] not in self.cycles:
                        self.cycles.append(path[path.index(m):] + [m])
                    continue
                _dfs(path + [m])

        for m in range(self.n_microservice):
            _dfs([m])

    def save_graph(self):
        with Application.graph_file.open("wb") as f_path:
            pickle.dump(self.adj, f_path)
            pickle.dump(self.G, f_path)
            pickle.dump(self.microservices, f_path)
            pickle.dump(self.n_microservice, f_path)

    def load_graph(self, is_draw=False, file_pth=None):
        print("====加载已生成的图，graph0文件存在：{}，历史生成图数：{}====".
              format(Application.graph_file.exists(),
                     len(list(Application.sv_dir.iterdir()))))
        try:
            graph_file_pth = Path(file_pth) if file_pth is not None else Application.graph_file
            with graph_file_pth.open("rb") as f_path:
                self.adj = pickle.load(f_path)
                self.G = pickle.load(f_path)
                self.microservices = pickle.load(f_path)
                self.n_microservice = pickle.load(f_path)
        except FileNotFoundError:
            f = list(Application.sv_dir.iterdir())
            if len(f) <= 0:
                raise FileNotFoundError
            f.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
            with f[-1].open("rb") as f_path:
                self.adj = pickle.load(f_path)
                self.G = pickle.load(f_path)
                self.microservices = pickle.load(f_path)
                self.n_microservice = pickle.load(f_path)
        if is_draw:
            self.draw_graph()

    def draw_graph(self):
        # 可视化图
        # 提取节点的颜色属性
        node_colors = [self.G.nodes[node]["color"] for node in self.G.nodes]

        # 使用Matplotlib绘制图形
        pos = nx.circular_layout(self.G)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=12,
            font_color="white",
            font_weight="bold",
            arrows=True,
        )
        edge_labels = {(u, v): d["weight"] for u, v, d in self.G.edges(data=True)}
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=15)

        # 提取边的注释信息
        edge_labels = {
            (u, v): data["comment"] for u, v, data in self.G.edges(data=True)
        }

        # 使用nx.draw_networkx_edge_labels添加边的注释标签
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=10)

        # node_names = {node: self.G.nodes[node]["name"] for node in self.G.nodes}
        # nx.draw_networkx_labels(
        #     self.G, pos, labels=node_names, font_size=10, font_color="black"
        # )  # 添加节点名称

        plt.title("1122")
        plt.show()


class Cluster:
    # 记录当前集群内所有的服务器，以id区分
    cluster = {}
    s_m = None

    # class Server:
    #     def __init__(self, app: Application, server_type=0, server_index=-1):
    #         # 服务器类型：0代表边缘服务器，1云服务器
    #         self.server_type = server_type
    #         self.limited_instance = np.random.randint(3, 15) \
    #             if server_type == 0 else float('inf')
    #         # 当前服务器上已部署的微服务：实例数
    #         self.svc2instance = {}
    #         self.server_index = server_index
    #         for m in app.microservices:
    #             self.svc2instance[m.name] = np.random.choice(
    #                 [0, 1], 1, p=[0.9, 0.1]).item()
    #             Cluster.s_m[server_index, m.name] = self.svc2instance[m.name]
    #
    #     def deploy_mcs(self, mcs: Microservice, num_ins: int):
    #         """
    #         在服务器上部署服务的实例
    #         Args:
    #             mcs: 部署的微服务
    #             num_ins: 实例数
    #         """
    #         self.svc2instance[mcs.name] = max(self.svc2instance.get(mcs.name, 0) + num_ins, 0)
    #         Cluster.s_m[self.server_index, mcs.name] = self.svc2instance[mcs.name]

    @staticmethod
    def init_cluster(n_servers, app: Application):
        Cluster.s_m = np.zeros((n_servers, len(app.microservices)))
        Cluster.s_m[-1] = 1
        # for index in range(n_servers - 1):
        #     Cluster.cluster[index] = Cluster.Server(app, server_index=index)
        # Cluster.cluster[-1] = Cluster.Server(app, 1, server_index=-1)

    @staticmethod
    def reset_cluster():
        Cluster.s_m = np.zeros_like(Cluster.s_m)
        Cluster.s_m[-1] = 1

    @staticmethod
    def deploy_mcs(server_ind, mcs_name: int, add_ins):
        min_svc = np.argmin(Cluster.s_m[server_ind])
        max_svc = np.argmax(Cluster.s_m[server_ind])
        if mcs_name == max_svc and Cluster.s_m[server_ind, max_svc] - Cluster.s_m[server_ind, min_svc] >= 10:
            Cluster.s_m[server_ind, max_svc] = min(Cluster.s_m[server_ind, max_svc],
                                                   Cluster.s_m[server_ind, max_svc] + add_ins)
            return
        if server_ind < Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 15 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1
        if server_ind == Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 100 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1

        Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name] + add_ins, 0)
        if server_ind == len(Cluster.s_m) - 1:
            Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name], 1)

    @staticmethod
    def deploy_mcs_mask(server_ind, mcs_name: int, add_ins, adj):
        svcNotIngraph = np.where(np.all(adj == 0, axis=1))[0].astype(np.int32)
        min_svc = np.argmin(Cluster.s_m[server_ind])
        max_svc = np.argmax(Cluster.s_m[server_ind])
        if mcs_name == max_svc and Cluster.s_m[server_ind, max_svc] - Cluster.s_m[server_ind, min_svc] >= 10:
            Cluster.s_m[server_ind, max_svc] = min(Cluster.s_m[server_ind, max_svc],
                                                   Cluster.s_m[server_ind, max_svc] + add_ins)
            return
        if server_ind < Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 15 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1
        if server_ind == Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 100 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1

        Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name] + add_ins, 0)
        if server_ind == len(Cluster.s_m) - 1:
            Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name], 1)
        for svc in svcNotIngraph:
            Cluster.s_m[:, svc] = 0

    # @staticmethod
    # def reset_cluster():
    #     Cluster.s_m = np.zeros_like(Cluster.s_m)
    #     Cluster.s_m[-1] = 1
    #     for k in Cluster.cluster:
    #         Cluster.cluster[k].svc2instance = {}
    #     for k in Cluster.cluster[-1].svc2instance:
    #         Cluster.cluster[-1].svc2instance[k] = 1

    def _init_edge(self):
        """
        初始化边缘服务器属性
        """
        self.server_type = 0
        self.limited_instance = np.random.randint(3, 15)
        self.memory = np.random.randint(400, 1100) * MB
        self.cpu = np.random.randint(1, 4)
        self.bandwidth = np.random.randint(100, 200) * MHZ

    def _init_cloud(self):
        """
        初始化云服务器属性
        """
        self.server_id = -1
        self.server_type = 1
        self.limited_instance = float('inf')
        self.memory = 1 * GB
        self.cpu = 4
        self.bandwidth = 2 * GB

    @staticmethod
    def remove_server(i):
        Cluster.s_m = np.concatenate([Cluster.s_m[:i, ...], Cluster[i + 1:, ...]])

    @staticmethod
    def add_server():
        new_server = np.zeros((Cluster.s_m.shape[0], Cluster.s_m.shape[1]))
        Cluster.s_m = np.concatenate([Cluster.s_m[:-1, ...], new_server,
                                      Cluster.s_m[-1, ...]])


class Microservice:
    def __init__(self, name):
        # 服务的索引
        self.name = name
        # 存储的是服务的索引
        self.children = []

    def get_prc_time(self, src_server, n_user):
        total_time = 0
        # 1.如果在目标服务器上存在目标服务的实例，则直接在目标服务器上处理；
        if Cluster.s_m[src_server, self.name] > 0:
            total_time += Metrics.process_time(n_user, instance=Cluster.s_m[src_server, self.name])
        else:
            # 2.否则遍历边缘服务器集，查找可处理的服务器；
            available_server = list(*np.where(Cluster.s_m[:-1, self.name] > 0))
            if len(available_server) > 0:
                total_time += Metrics.process_time(n_user,
                                                   instance=Cluster.s_m[available_server[0], self.name])
            # 3.没有则在云服务器上处理
            # 云服务器的来回传播延迟
            else:
                t_prop = 2 * torch.nn.init.trunc_normal_(tensor=torch.zeros(1, ), mean=11, std=2, a=10, b=15).item()
                if Cluster.s_m[-1, self.name] <= 0:
                    Cluster.s_m[-1, self.name] = 1
                total_time += t_prop + Metrics.process_time(n_user, instance=Cluster.s_m[-1, self.name])
        return total_time


class UserGroup:
    users = {}

    class User:
        def __init__(self, longitude=-1, latitude=-1, server=-1):
            # 代表此用户不在任一边缘服务器附近，请求将直接被调度到云服务器
            self.location = -1
            self.longitude = longitude
            self.latitude = latitude
            # 表示用户当前处于哪个服务器范围，-1 代表在云服务器范围
            self.server = server
            UserGroup.users[server] = UserGroup.users.get(server, [])
            UserGroup.users[server].append(self)

    def __init__(self):
        pass

    @staticmethod
    def init_users(n_users: int = 200):
        # for _ in range(n_users):
        #     server = np.random.choice(list(Cluster.cluster.keys()), 1).item()
        #     UserGroup.User(server=server)
        pass

    def send_request(self, mcs, n_user, app: Application, src_server_id=-1, path=None, mcs_response_time=None):
        return self.get_process_time(mcs, n_user, app, src_server_id, path, mcs_response_time=mcs_response_time)

    def get_process_time(self, mcs: Microservice, n_user, app: Application, src_server_id=-1, path=None, sub_path=None,
                         mcs_response_time=None):
        assert mcs_response_time is not None and isinstance(mcs_response_time, dict), "must pass dict mcs_response_time"
        if sub_path is None:
            sub_path = []
        if path is None:
            path = []
        sub_path.append(mcs.name)
        prc_time = 0
        # 遍历到度为0的节点，叶子节点，记录该子调用路径
        if len(mcs.children) == 0:
            path.append(sub_path)
        for m in mcs.children:
            # prc_time += self.get_process_time(app.microservices[m], n_user, app, src_server_id, path, [] + sub_path,
            #                                   mcs_response_time=mcs_response_time)
            prc_time = max(prc_time,
                           self.get_process_time(app.microservices[m], n_user, app, src_server_id, path, [] + sub_path,
                                                 mcs_response_time=mcs_response_time))
        cur_mcs_time = mcs.get_prc_time(src_server_id, n_user)
        prc_time += cur_mcs_time
        mcs_response_time[mcs.name] = mcs_response_time.get(mcs.name, []) + [cur_mcs_time]
        return round(prc_time, 3)


class EdgeCloudSim:
    traffic_pth = Path(cur_dir, "dataset", "traffic_data_simulation.npy")
    data_mcs_pth = Path(cur_dir, "dataset", "traffic_mcs_data_simulation.npy")
    traffic_mcs_pth = Path(cur_dir, "dataset", "traffic_data_simulation.npz")
    data_pics_dir = Path(cur_dir, "dataset", "pics", 'mcs_traffic')
    data_pics_dir.mkdir(parents=True, exist_ok=True)
    [i.unlink() for i in data_pics_dir.iterdir()]

    # 模型训练文件保存路径完全性检查：1.训练文件保存根目录 2.gma训练文件保存目录 3.模型参数保存目录 4.模型对象保存文件
    train_data_pth = Path(gma_model, "训练数据")
    train_data_pth.mkdir(parents=True, exist_ok=True)
    gma_train_pth = Path(train_data_pth, "GMA数据")
    gma_train_pth.mkdir(parents=True, exist_ok=True)

    def __init__(self, n_server=8, n_user=200, is_gen=False, if_mask=False, adj_construct=None, is_init_app=True):
        self.is_continuous = False
        self.mode = "train"
        self.algotype = "gat"
        if is_init_app:
            self.application = Application(is_gen, adj_construct=adj_construct)
        self.num_mcs = len(self.application.microservices)
        self.n_server = n_server
        self.n_user = n_user
        self.t_s_m = np.load(EdgeCloudSim.data_mcs_pth)[:, :self.n_server, :]
        self.t_s_m[..., -2] += 300
        self.traffic = np.array(([int(i.strip()) for i in list(Path(self.get_dataset_dir(),
                                                                    "flows.txt").open("r"))]))
        self.reward_pth = Path(EdgeCloudSim.gma_train_pth, "gma_reward.pt")
        self.train_state_pth = Path(EdgeCloudSim.gma_train_pth, "gma_object.pt")
        self.t = 0
        self.action_set = [-1, 0, 1]
        self.actions = self.get_actions()
        self.if_mask = if_mask
        self.action_space = np.arange(-2, 3)
        self.entrance_mcs = np.load("/Users/username/Project/Paper-Scalable/simulation/dataset/access_data.npy")
        self.actual_workload = compute_disseminated_workload(self.application.adj,
                                                             self.entrance_mcs,
                                                             self.t_s_m)
        self.msc_edge_adj = compute_mcs_edge_adj(self.application.adj,
                                                 self.entrance_mcs,
                                                 self.t_s_m)
        Cluster.init_cluster(n_server, self.application)
        UserGroup.init_users(self.n_user)

    def get_actions(self):
        def _get_actions(d=self.num_mcs):
            re = []
            if d == 0:
                return []
            for i in self.action_set:
                acts = _get_actions(d - 1)
                if len(acts) > 0:
                    re.extend([[i] + j for j in acts])
                else:
                    re.append([i])
            return re

        return _get_actions()

    def get_dataset_dir(self):
        return Path(os.path.dirname(__file__), "dataset")

    def foward_request_gen(self, repetition=1):
        """
        模拟repetition次集群用户请求，即某一时刻内，集群内用户请求应用
        Args:
            repetition: 模拟次数
        """
        statistics_res = []
        for _ in range(repetition):
            res_seq = []
            for s, users in UserGroup.users.items():
                user2msc = {}
                # 生成一个服务器内用户对服务的访问分布
                mcs = np.random.choice(self.application.microservices, len(users))
                user2msc[mcs] = user2msc.get(mcs, 0) + 1
                for mcs, n_user in user2msc.items():
                    path = []
                    res = UserGroup().send_request(mcs, n_user, self.application, s, path)
                    _path = sorted(set([tuple(i) for i in path]), key=lambda i: path.index(list(i)))
                    print("访问用户数{},是否有重复路径:{},响应时间为{}ms，路径条数：{}, 路径长度：{}, 访问路径为：{}"
                          .format(n_user, not len(path) == len(_path), res, len(path),
                                  [len(i) for i in path], path))
                    res_seq.append(res * n_user)
            # 总请求时间/用户数
            avg_res = np.sum(res_seq) / np.sum([len(i) for i in UserGroup.users.values()])
            print("平均请求响应时间：{}".format(avg_res))
            print("===================")
            statistics_res.append(avg_res)

        statistics_res = MinMaxScaler().fit_transform(np.array(statistics_res).reshape(-1, 1))
        return np.array(statistics_res).squeeze(-1)

    def send_dataset_workload(self, t=0):
        """
        按数据集发送流量，同时获取环境反馈的状态
        @param t: 当前实验的时间步
        @return: 集群的平均响应时间（加权计算），各服务器的响应时间（加权），集群的平均响应时间，各服务器的响应时间
        """
        # user request response time
        user_res = np.zeros((self.n_server, self.num_mcs))
        total_res = np.zeros((self.n_server, self.num_mcs))
        # Response times of m services on n servers at time t
        svc_res = np.zeros((self.n_server, self.num_mcs))
        total_users = 0
        redist_traffic = self.t_s_m[t] + 200
        # redist_traffic = redistribute_traffic(self.t_s_m[t] + 400, Cluster.s_m, 90)

        for s in range(self.n_server):
            mcs_response_time = {}
            entry_service = np.where(self.entrance_mcs[t, s] == 1)[0]
            for m in entry_service:
                mcs = self.application.microservices[m]
                path = []
                n_mcs_user = redist_traffic[s, m]
                total_users += n_mcs_user
                if s == self.n_server - 1:
                    s = -1
                res = UserGroup().send_request(mcs, n_mcs_user, self.application, s, path, mcs_response_time)
                _path = sorted(set([tuple(i) for i in path]), key=lambda i: path.index(list(i)))
                # logger.info("服务器{},服务{},访问用户数{},是否有重复路径:{},响应时间为{}ms，路径条数：{}, 路径长度：{}, 访问路径为：{}"
                #             .format(s, mcs.name, n_mcs_user, not len(path) == len(_path), res, len(path),
                #                     [len(i) for i in path], path))
                total_res[s, m] = res * n_mcs_user
                user_res[s, m] = res  # 每个服务器入口服务的平均响应时间
            for i in range(svc_res.shape[-1]):
                svc_res[s, i] = np.mean(mcs_response_time.get(i, 0))
        # 总请求时间/用户数
        avg_res = np.round(np.sum(total_res) / total_users)
        avg_user_res = np.mean(user_res)
        print("平均请求响应时间：{}".format(round(avg_user_res)))
        print("===================")
        return avg_res, total_res, avg_user_res, user_res, svc_res

    def traffic(self, slot=200):
        period = 10
        users2mcs = {}
        for t in range(slot):
            for m in self.application.microservices:
                users2mcs[m] = users2mcs.get(m, [])
                if t == 0:
                    users2mcs[m].append(20)
                else:
                    if (t // period) & 1 == 0:
                        print(t // period)
                        users2mcs[m].append(users2mcs[m][-1] + np.random.randint(1, 10))
                    else:
                        users2mcs[m].append(users2mcs[m][-1] + np.random.randint(-9, -1))
        return users2mcs

    def get_state(self, t):
        """
        构造特征，当前微服务的负载、实例数，当前服务器已部署的实例数、负载

        Args:
            t: current time step

        Returns:
            numpy array:  server obs, svc obs
        """
        m = self.num_mcs
        s = self.n_server
        user_server = np.sum(self.t_s_m[t], axis=-1, keepdims=True)
        actual_workload = self.actual_workload[t, :s, :m]
        avg_total_res, total_res, avg_user_res, user_res, svc_res = self.send_dataset_workload(t)
        avg_res_all_servers = np.repeat(np.array(avg_user_res)[np.newaxis, np.newaxis], s, axis=0)
        workload_on_servers = np.concatenate([self.t_s_m[t], np.zeros([self.t_s_m[t].shape[0], m - 5])], axis=-1)
        server_obs = np.concatenate([workload_on_servers, Cluster.s_m, avg_res_all_servers,
                                     user_res, svc_res, actual_workload], axis=-1)

        workload_on_svc = actual_workload.T
        total_svc_workload = np.sum(workload_on_svc, axis=-1, keepdims=True)
        avg_svc_workload = np.mean(workload_on_svc, axis=-1, keepdims=True)
        total_svc_ins = np.sum(Cluster.s_m.T, axis=-1, keepdims=True)
        avg_svc_ins = np.mean(Cluster.s_m.T, axis=-1, keepdims=True)
        avg_svc_res = np.mean(svc_res.T, axis=-1, keepdims=True)
        svc_obs = np.concatenate([total_svc_workload, avg_svc_workload, total_svc_ins, avg_svc_ins, avg_svc_res],
                                 axis=-1)

        return server_obs, svc_obs, user_res

    @staticmethod
    def state_factorization(state, m, s):
        """
        状态向量解构
        Args:
            state: 状态向量
            m: 微服务数量
            s: 服务器数量

        Returns: 每个微服务的总实例数、分布实例数，每个微服务的总负载、分布负载，服务器状态

        """
        server_state = state[:, :3 * m + 2]
        mcs_res = state[:, 3 * m + 2:4 * m + 2]
        mcs_ins = state[:, 4 * m + 2:5 * m + 2]
        mcs_resp_server = state[:, 5 * m + 2:5 * m + 2 + m * s]
        mcs_insp_server = state[:, 5 * m + 2 + m * s:]
        return mcs_ins, mcs_insp_server, mcs_res, mcs_resp_server, server_state

    def reward(self, user_res, t=0):
        server_res = np.mean(user_res, axis=-1)
        server_ins = np.sum(Cluster.s_m, axis=-1)
        server_reward = []
        sla = 50
        limit_rsc_edge = 8
        cloud_rsc_limit = 20

        print("instance distribution: {}".format(Cluster.s_m))
        # print("accessed services:".format(self.entrance_mcs[t]))
        # print("workload distribution: {}".format(self.actual_workload[t]))

        for i in range(self.n_server):
            res_time = server_res[i]
            total_ins = server_ins[i]
            reward_res = sla - res_time
            limit_rsc = cloud_rsc_limit if i == self.n_server - 1 else limit_rsc_edge
            reward_ins = -1 * total_ins
            workload_on_server = self.actual_workload[t, i][np.newaxis, :]
            resource_pattern = Cluster.s_m[i][np.newaxis, :]
            pearsonr_score = pearsonr(self.actual_workload[t, i], Cluster.s_m[i])[0]
            cosine_score = cosine_similarity(workload_on_server, resource_pattern).item()
            pattern_score = 0.7 * cosine_score + 0.3 * pearsonr_score if not np.isnan(pearsonr_score) else cosine_score

            # server_reward.append(0.12 * reward_ins + 0.75 * reward_res)
            server_reward.append(0.4 * reward_ins + 0.6 * reward_res)

        return np.array(server_reward)

    def deploy_mcs(self, server_ind, mcs_name: int, add_ins):
        num_server = len(Cluster.s_m)
        # Set resource thresholds for edge servers and cloud servers.
        total_rsc = 50 if server_ind < num_server - 1 else 200
        used_rsc = np.sum(Cluster.s_m[server_ind])
        remain_rsc = max(total_rsc - used_rsc, 0)
        if add_ins > 0 >= remain_rsc:
            return
        Cluster.s_m[server_ind, mcs_name] += add_ins
        Cluster.s_m[server_ind, mcs_name] = max(0, Cluster.s_m[server_ind, mcs_name])

    def reset(self):
        Cluster.reset_cluster()
        obs, svc_obs, user_res = self.get_state(0)
        agent_id = np.reshape(np.arange(0, self.n_server), (1, -1, 1))
        adj = np.ones((self.n_server, self.n_server))
        return obs[np.newaxis, ...], svc_obs, agent_id, obs[:, np.newaxis, :].repeat(self.n_server, axis=1), adj

    def step(self, actions: Union[torch.Tensor, np.ndarray], t=0):
        action_step = self.action_space[actions]
        original_shape = action_step.shape
        action_step_resize = np.reshape(action_step, (-1, original_shape[-2], original_shape[-1]))
        for i in range(action_step_resize.shape[0]):
            for j in range(action_step_resize.shape[1]):
                for k in range(action_step_resize.shape[2]):
                    self.deploy_mcs(j, k, action_step_resize[i, j, k])
        obs, svc_obs, user_res = self.get_state(t)
        agent_id = np.reshape(np.arange(0, self.n_server), (1, -1, 1))
        adj = np.ones((self.n_server, self.n_server))
        dones = np.full((1, self.n_server), t >= 199)
        return (obs, svc_obs, agent_id, obs, adj,
                self.reward(user_res, t)[np.newaxis, :, np.newaxis], dones, None)

    def step1(self, actions: torch.Tensor, t=0):
        """
        执行给定的动作集合
        Args:
            t:
            actions: (s, m)

        Returns:
            tuple: 状态集合，奖励
        """
        # actions = np.array([self.actions[i.item()] for i in actions.type(torch.int16)])
        print("当前欲执行动作:{}{}资源分布:{}".format(actions, os.linesep, Cluster.s_m))
        for i in range(actions.shape[0]):
            for j in range(actions.shape[1]):
                # print("第{}个服务器上第{}个服务实例数{}，动作{}".format(i, j, Cluster.s_m[i, j], actions[i, j]))
                self.deploy_mcs(i, j, actions[i, j])
                # Cluster.cluster[i].deploy_mcs(self.application.microservices[j], action_mcs)
                # print("修改完后{}".format(Cluster.s_m[i, j]))
                pass
        state = self.get_state(t)
        agent_id = np.reshape(np.arange(0, self.n_server), (1, -1, 1))
        adj = np.ones((self.n_server, self.n_server))
        dones = np.full((1, self.n_server), t < 199)
        return state, agent_id, state, adj, self.reward(state), dones, None

def _t2n(x):
    return x.detach().cpu().numpy()


class GMPERunner:
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, num_servers=8, adj_construct=None):
        self.n_rollout_threads = 1
        self.envs = EdgeCloudSim(n_server=num_servers, is_gen=True, adj_construct=adj_construct)
        num_services = self.envs.application.adj.shape[0]
        self.svc_adj = self.envs.application.adj[np.newaxis, ...].repeat(num_servers, axis=0)
        self.num_env_steps = 10000
        self.episode_length = 200
        self.all_args = None
        self.num_agents = num_servers
        self.num_services = num_services
        self.use_linear_lr_decay = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = GRMAPPOPolicy(5 * num_services + 1, num_services, self.svc_adj, n_servers=num_servers)
        self.trainer = GR_MAPPO(self.policy, device=self.device)
        self.buffer = GraphReplayBuffer(5 * num_services + 1, self.num_agents, self.num_services)
        self.recurrent_N = 1
        self.use_centralized_V = True
        self.hidden_size = 64
        self.save_dir = "/Users/username/Project/Paper-Scalable/simulation/ICSE2026模型保存/依赖自学习-应用2"
        self.training_log_dir = Path(self.save_dir, "training_log")
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.restore()

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        try:
            policy_actor_state_dict = torch.load(
                str(self.save_dir) + "/actor.pt", map_location=torch.device("cpu")
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
        except FileNotFoundError:
            print("actor model not saved")
        # try:
        #     policy_critic_state_dict = torch.load(
        #         str(self.save_dir) + "/critic.pt", map_location=torch.device("cpu")
        #     )
        #     self.policy.critic.load_state_dict(policy_critic_state_dict)
        # except FileNotFoundError:
        #     print("critic model not saved")

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def run(self):
        self.warmup()

        start_time = time.time()
        episodes = 500
        rewards_log = []
        # This is where the episodes are actually run.
        for episode in range(episodes):
            print("Episode {}/{}".format(episode + 1, episodes))
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            rewards_episode = []
            # Reset the environment at the beginning.
            self.envs.reset()
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obs reward and next obs
                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(actions_env)
                )
                rewards_episode.append(rewards)

                data = (
                    obs,
                    svc_obs,
                    agent_id,
                    node_obs,
                    adj,
                    self.envs.msc_edge_adj[step + 1],
                    agent_id,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            rewards_log.append(np.array(rewards_episode).sum())

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                    (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % 5 == 0 or episode == episodes - 1:
                self.save()
                torch.save(np.array(rewards_log), Path(self.training_log_dir, "rewards.pt"))

            # # log information
            # if episode % self.log_interval == 0:
            #     end = time.time()
            #
            #     env_infos = self.process_infos(infos)
            #
            #     avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
            #     train_infos["average_episode_rewards"] = avg_ep_rew
            #     print(
            #         f"Average episode rewards is {avg_ep_rew:.3f} \t"
            #         f"Total timesteps: {total_num_steps} \t "
            #         f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
            #     )
            #     self.log_train(train_infos, total_num_steps)
            #     self.log_env(env_infos, total_num_steps)
            #
            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)
        end_time = time.time()
        print("training done, total time taken:", format_training_duration(end_time - start_time))

    def warmup(self):
        # reset env
        obs, svc_obs_, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
        share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
        share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
            self.num_agents, axis=1
        )
        svc_obs = svc_obs_[np.newaxis, np.newaxis, ...].repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.svc_obs[0] = svc_obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.svc_adj[0] = self.envs.msc_edge_adj[0].copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.node_obs[step]),
                                            np.concatenate(self.buffer.adj[step]),
                                            np.concatenate(self.buffer.agent_id[step]),
                                            np.concatenate(self.buffer.share_agent_id[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            svc_obs=np.concatenate(self.buffer.svc_obs[step]),
                                            svc_adj=np.concatenate(self.buffer.svc_adj[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions,
        )

    def insert(self, data):
        (
            obs,
            svc_obs,
            agent_id,
            node_obs,
            adj,
            svc_adj,
            agent_id,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, svc_obs=svc_obs, svc_adj=svc_adj)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     np.concatenate(self.buffer.node_obs[-1]),
                                                     np.concatenate(self.buffer.adj[-1]),
                                                     np.concatenate(self.buffer.share_agent_id[-1]),
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]),
                                                     np.concatenate(self.buffer.svc_obs[-1]),
                                                     np.concatenate(self.buffer.svc_adj[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    # @measure_performance
    def evaluate_method(self):
        """
        这个函数可用于所提方法各种用途的验证，包括：环境规模变化下的适应性验证，流量适应性验证
        """
        print(f"环境中服务器数量{self.envs.n_server}，服务数量{self.envs.application.n_microservice}")
        start_time = time.time()
        process = psutil.Process(os.getpid())

        start_mem = process.memory_info().rss
        eval_rnn_states = np.zeros(
            (1, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (1, self.num_agents, 1), dtype=np.float32
        )
        obs, svc_obs, agent_id, node_obs, adj = self.envs.reset()
        res_per_group = []
        pods_per_group = []
        pattern_score = []

        for times in range(5):
            res_cur = []
            pods_cur = []
            pattern_score_cur = []
            for eval_step in range(100):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states, exe_time, exe_mem = self.trainer.policy.act(
                    np.concatenate(obs),
                    node_obs,
                    np.ones((self.num_agents, self.num_agents, self.num_agents)),
                    np.concatenate(agent_id),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    svc_obs=svc_obs[np.newaxis, :].repeat(self.num_agents, 0),
                    svc_adj=self.envs.msc_edge_adj[eval_step]
                )
                end_time = time.time()
                end_mem = process.memory_info().rss
                # print("全程所耗费的时间: {:.2f},内存:{:.2f}".format((end_time - start_time) * 1000,
                #                                                     (end_mem - start_mem) / (1024 * 1024)))

                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(eval_action)
                )
                workload_on_server = np.sum(self.envs.actual_workload[eval_step], 0, keepdims=True)
                mask = workload_on_server != 0
                workload_on_server[mask] = workload_on_server[mask] / np.min(workload_on_server)
                resource_pattern = np.sum(Cluster.s_m, 0, keepdims=True)
                # cosine_score = cosine_similarity(workload_on_server, resource_pattern).item()
                cosine_score = 0
                pattern_score_cur.append(cosine_score)
                res_cur.append(self.envs.send_dataset_workload(eval_step)[2])
                pods_cur.append(Cluster.s_m.sum())
                obs = np.array(obs)[np.newaxis, ...]
                node_obs = np.array(node_obs)[np.newaxis, ...].repeat(self.num_agents, 0)
                eval_rnn_states = np.array(
                    np.split(_t2n(eval_rnn_states), 1)
                )
                eval_rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (1, self.num_agents, 1), dtype=np.float32
                )
                eval_masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )
            res_per_group.append(res_cur)
            pods_per_group.append(pods_cur)
            pattern_score.append(pattern_score_cur)
        # return exe_time, exe_mem
        # return end_time - start_time, (end_mem - start_mem) / (1024 * 1024)
        res_per_group = np.array(res_per_group)
        pods_per_group = np.array(pods_per_group)
        pattern_score = np.array(pattern_score)
        results_save_dir = Path("/Users/username/Project/Paper-Scalable/simulation/ICWS2025实验结果/所提方法结果")
        results_save_dir.mkdir(parents=True, exist_ok=True)
        results_save_file = Path(results_save_dir, "【迁移性实验】服务器35.npz")
        np.savez(
            results_save_file,
            res=res_per_group,
            pods=pods_per_group,
            pattern_score=pattern_score)
        print("响应时间: {}, 实例数: {}".format(res_per_group.mean(), pods_per_group.mean()))
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(res.shape[-1]), 7), y=res.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(pods.shape[-1]), 7), y=pods.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # sns.lineplot(x=np.tile(np.arange(pattern_score.shape[-1]), 7), y=pattern_score.flatten(), label="Response",
        #              errorbar="se",
        #              legend=False, color="b", linewidth=1.1)
        # plt.show()
        #
        # plt.figure()
        # res_distribution = np.mean(res, 0)
        # res_sorted = np.sort(res_distribution)
        # cdf = np.arange(1, len(res_sorted) + 1) / len(res_sorted)
        # plt.plot(res_sorted, cdf, marker='.', linestyle='none')
        # plt.xlabel('响应时间')
        # plt.ylabel('累积分布 (CDF)')
        # plt.title('响应时间的累积分布图')
        # plt.grid(True)
        # plt.savefig(
        #     "/Users/username/Project/Paper-Scalable/simulation/实验中间结果/结果图/CMA响应时间累积分布图.png")
        # plt.show()


def calculate_improvement(algorithm_A, algorithm_B):
    improvements = []
    for a, b in zip(algorithm_A, algorithm_B):
        improvement = ((b - a) / b) * 100  # 计算相对优胜百分比
        improvements.append(improvement)
    return improvements


def plot_reward_curve(file_path):
    # 读取 CSV 文件
    data = torch.load(file_path)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制奖励曲线
    ax.plot(data, label='Reward', color='blue', lw=2)

    # 设置标题
    ax.set_title('Reward Curve', fontsize=14)

    # 设置 X 和 Y 轴标签
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)

    # 保留 X 和 Y 轴刻度线
    ax.tick_params(axis='both', direction='in', length=6, width=1.5, colors='black')  # 设置刻度线的方向、长度和颜色

    # 去除背景网格线
    ax.grid(False)

    # 设置黑色箭头样式的 X 和 Y 轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')

    # 设置箭头
    ax.annotate('', xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<|-', color='black', lw=1.5), xycoords='axes fraction',
                textcoords='axes fraction')
    ax.annotate('', xy=(0, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<|-', color='black', lw=1.5), xycoords='axes fraction',
                textcoords='axes fraction')

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    adj_matrix = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    noise_adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    istio_adj_mat = np.array([[0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0]])
    noise_istio_adj_mat = np.array([[0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 1],
                                    [0, 1, 0, 0, 0]])

    runner = GMPERunner(num_servers=8, adj_construct=adj_matrix)
    runner.run()
    runner.evaluate_method()