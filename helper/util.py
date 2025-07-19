import copy
import logging
import os
import pickle
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from logging import handlers
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import numpy as np

KUBERNETES_NAMESPACE = "default"


def get_project_root():
    path = os.path.abspath(os.curdir).split(os.sep)
    ind = -1
    for i in range(len(path)):
        if re.findall("Edge", path[i]):
            ind = i
            break
    return os.sep.join(path[:ind + 1])


def get_logger():
    root_dir = get_project_root()
    log_path = os.sep.join([root_dir, "log"])
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s: %(message)s')
    func_logger = logging.getLogger(__file__)
    handler = handlers.RotatingFileHandler(os.sep.join([log_path, os.path.basename(__file__).split(".")[0] + ".log"]),
                                           encoding='utf-8',
                                           maxBytes=104857600, backupCount=2)
    fomatter = logging.Formatter('%(asctime)s- %(levelname)s: %(message)s')
    handler.setFormatter(fomatter)
    func_logger.addHandler(handler)
    return func_logger


logger = get_logger()


def smooth(x, window=1):
    r = []
    if len(x) == 0 or len(x) == 1:
        return x
    for i in range(len(x)):
        last = 0.2 * x[i - 1] if i >= 1 else 0.1 * x[1]
        r.append(last + 0.8 * x[i])
    return r


def pods_is_ready(tolerance=30):
    """
    执行kubectl命令，查看当前pod是否处于running状态

    """

    def is_ready():
        log_path = Path.joinpath(Path.home(), "pods_status.yaml")
        os.system("kubectl get pods -n {} -o wide > {}".format(KUBERNETES_NAMESPACE, log_path))
        with Path.open(log_path) as f:
            f.readline()
            for line in f:
                if line.split()[2] != 'Running':
                    logger.info("{}没有准备好！".format(line.split()[0]))
                    return False
            else:
                return True

    st = datetime.now()
    timedel = timedelta()
    while timedel < timedelta(seconds=tolerance) and not is_ready():
        time.sleep(3)
        timedel = datetime.now() - st
    logger.info("等待 pod 部署完成，耗时{}秒，{}全部部署完成".format(timedel.seconds, "已" if timedel < timedelta(
        seconds=tolerance) else "未"))
    return timedel.seconds


def print_pods_state():
    """
    查询各个节点上的pod运行信息

    :return:
    """
    log_path = Path.joinpath(Path.home(), "pods_info.out")
    os.system("kubectl get pods -n {} -o wide > {}".format(KUBERNETES_NAMESPACE, log_path))
    pods_stat = {}
    worker_nodes = get_nodes()
    with Path.open(log_path) as f:
        f.readline()
        for cur_line in f:
            elements = cur_line.split()
            # 这一步是为了获取这个pod对应的service name
            pod_name = "-".join(elements[0].split("-")[:2])
            node = None
            status = 0
            for e in elements:
                if e in worker_nodes:
                    node = e
                if e == "Running":
                    status = 1
            # 如果pod已经起来，且部署在节点node上，则做统计
            if status == 1 and node:
                if node not in pods_stat:
                    pods_stat[node] = {}
                pods_stat[node][pod_name] = pods_stat[node].get(pod_name, 0) + 1

    # 这一步是容错处理，防止node上没有包含全部服务，后续读取错误，没有的一律置0
    for node in worker_nodes:
        if node not in pods_stat:
            pods_stat[node] = {}
        for svc in ["productpage-v1", "details-v1", "ratings-v1", "reviews-v1", "reviews-v2", "reviews-v3"]:
            if svc not in pods_stat[node]:
                pods_stat[node][svc] = 0

    return pods_stat


def talley_pods_replicas():
    log_path = Path.joinpath(Path.home(), "pods_info.out")
    os.system("kubectl get pods -n {} -o wide > {}".format(KUBERNETES_NAMESPACE, log_path))
    svc = {}
    svc_running = {}
    svc_others = {}
    with Path(log_path).open() as f:
        f.readline()
        for cur_line in f:
            elements = cur_line.split()
            # 这一步是为了获取这个pod对应的service name
            pod_name = "-".join(elements[0].split("-")[:2])
            svc[pod_name] = svc.get(pod_name, 0) + 1
            # pod状态检查
            for e in elements:
                if e == 'Running':
                    svc_running[pod_name] = svc_running.get(pod_name, 0) + 1
                    break
            else:
                svc_others[pod_name] = svc_others.get(pod_name, 0) + 1
    return svc, svc_running, svc_others


def fetch_res():
    """
    根据gatling生成的log文件计算响应时间，返回的是所有用户的平均等待时间

    :param root: log文件的根目录
    """
    threshold = 1000
    gatling_home = [i for i in Path.iterdir(Path.home())
                    if i.name.startswith("gatling")][0]
    results_path = Path.joinpath(gatling_home, "results")
    record_dir = Path.joinpath(Path.home(), "request_stats")
    if not Path.exists(record_dir):
        Path.mkdir(record_dir)
    sv_pth = Path.joinpath(record_dir, str(datetime.now()))
    subdirs = [subdir for subdir in Path.iterdir(results_path)
               if subdir.name.find('-') != -1]
    subdirs.sort(key=lambda x: eval(x.name.split('-')[-1]))
    logger.info("当前results下统计目录数量：{}".format(len(subdirs)))
    requests = []
    errors = 0
    total = 0
    for child_dir in subdirs:
        with Path.open(Path.joinpath(child_dir, "simulation.log")) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                total += 1
                elements = line.split()
                if elements[0].startswith('RE'):
                    try:
                        requests.append(eval(elements[-2]) - eval(elements[-3]))
                    except:
                        logger.error("不属于请求")
                        errors += 1
                        pass
        shutil.rmtree(child_dir)

    # 使用numpy保存的时候，不需要将保存的数据手动转换成numpy数组，包括基本类型
    np.savez(file=sv_pth, requests=requests,
             errors=errors,
             totals=total,
             n_sim=len(subdirs))

    opt_req = np.mean(requests)
    return opt_req


def calculate_distance_lat(latitude1, longitude1, latitude2, longitude2):
    latitude1, longitude1, latitude2, longitude2 = map(radians, [latitude1, longitude1, latitude2, longitude2])
    diff_lat = latitude1 - latitude2
    diff_lont = longitude1 - longitude2
    a = sin(diff_lat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(diff_lont / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371
    return round(distance, 2)


def geodistance(lng1, lat1, lng2, lat2):
    """

    :param lng1:
    :param lat1:
    :param lng2:
    :param lat2:
    :return:
    """
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 2)
    return distance


def get_system_path(*paths):
    return os.sep.join(paths)


def get_nodes():
    pth = Path.joinpath(Path.home(), "nodes.yaml")
    os.system("kubectl get nodes -o wide > {}".format(pth))
    nodes = []
    with Path.open(pth) as f:
        for line in f:
            nodes.append(line.split()[0])
    try:
        os.remove(pth)
    except FileNotFoundError:
        pass
    else:
        print("移除成功！" + pth.name)
    return nodes[1:]


def prepare_test_data(scale, data_scale=200):
    path_flow = Path(get_project_root(), "dataset", "flows.txt")
    flow_pic_dir = Path(get_project_root(), "trace_data", "流量数据")
    flow_pic_dir.mkdir(exist_ok=True)
    with Path.open(path_flow) as f:
        flows = [int(i) for i in f]
    test = flows[800:]
    test_data = copy.deepcopy(test)
    if data_scale < len(test_data):
        test_data = test_data[:data_scale]
    else:
        nt = data_scale - len(test)
        st = random.randint(0, 799)
        for ptr in range(nt):
            req = flows[(ptr + st) % len(flows)] + random.randint(-5, 40)
            if req <= 0:
                req = 40
            test_data.append(req)
    if scale != 1:
        test_data = [i * scale for i in test_data]
    with Path(flow_pic_dir, "{}倍流量".format(scale)).open("wb") as f:
        logger.info("测试数据准备完毕！一共{}条，源测试数据为{}条".format(len(test_data), len(test)))
        pickle.dump(test_data, f)
    return test_data


def send_req(req_num):
    """
    用于发送流量，并发数由参数指定，适用于测试场景
    """
    if not Path.exists(Path.joinpath(Path.home(), "send_request.sh")):
        raise Exception("发送流量脚本不存在")
    logger.info("开始发送流量，此刻的用户数为: {}".format(req_num))
    subprocess.call(["sh", Path.joinpath(Path.home(), "send_request.sh"),
                     str(req_num)])


def encode_action(n):
    """
    对动作进行编码，以便适用于机器学习算法

    :param n: 服务数
    :return: 动作编码
    """
    if n == 0:
        return []
    encode = []
    for i in range(-1, 2):
        result = encode_action(n - 1)
        if result:
            encode.extend([[i] + j for j in result])
        else:
            encode.append([i])
    return encode


def get_sla_violation(res_seq, sla):
    """
    计算sla违约率，计算方式：违约时间/执行总时间
    :param res_seq: 请求时间序列
    :param sla: 定义的sla
    """
    # print(sum([1 if i > sla else 0 for i in res_seq]), len(res_seq))
    return round(sum([1 if i > sla else 0 for i in res_seq]) * 100 / len(res_seq), 2)


def percent_growth(a, b):
    return 1
    # return round((a - b) * 100 / b, 2)


if __name__ == '__main__':
    get_logger()
pass
