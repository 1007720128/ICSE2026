import copy
import numpy as np
from networkx.algorithms.isomorphism.ismags import intersect
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn


def calculate_sla_violation_rate(response_times, sla_threshold=40):
    """
    Calculate the SLA violation rate

    Parameters:
    response_times (array-like): Sequence of response times
    sla_threshold (float): SLA response time threshold

    Returns:
    float: SLA violation rate
    """
    response_times = np.array(response_times)  # Convert to numpy array for convenience
    violation_rate_multi = []
    for i in range(len(response_times)):
        violations = np.sum(response_times[i] > sla_threshold)  # Count the number of violations
        violation_rate = violations / len(response_times[i])
        violation_rate_multi.append(violation_rate)  # Calculate the violation rate

    return np.array(violation_rate_multi)


def calculate_p95_latency(response_times, q=95):
    """
    Calculate the 95th percentile latency (P95 latency)

    Parameters:
    response_times (array-like): Sequence of response times

    Returns:
    float: P95 latency
    """
    p95_array = []
    for i in range(len(response_times)):
        p95_latency = np.percentile(response_times[i], q)  # Calculate the 95th percentile latency
        p95_array.append(p95_latency)
    return np.array(p95_array)


def format_training_duration(duration_seconds):
    days = duration_seconds // (24 * 60 * 60)
    hours = (duration_seconds % (24 * 60 * 60)) // (60 * 60)
    minutes = (duration_seconds % (60 * 60)) // 60
    seconds = duration_seconds % 60

    if days > 0:
        return f"{days} days {hours} hours {minutes} minutes {seconds} seconds"
    elif hours > 0:
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} minutes {seconds} seconds"
    else:
        return f"{seconds} seconds"


def evaluation_score(adj, resouce, workload, alpha=0.5, beta=0.5):
    """
    Calculate the evaluation score combining cosine similarity and weighted correlation.
    """

    return 0


def compute_mcs_edge_adj(adj_matrix, entry_services, traffic):
    """
    Calculating edge features between microservices.
    :param adj_matrix:
    :param entry_services:
    :param traffic:
    :return:
    """
    t, n, m = entry_services.shape
    t1, n, m = traffic.shape
    time = min(t, t1)
    result = np.zeros((time, n, adj_matrix.shape[0], adj_matrix.shape[0]))

    def dfs(path, t, n, transfer_traffic):
        # The dependency graph is guaranteed to be loop-free
        for child in range(m):
            if adj_matrix[path[-1], child] == 1:
                if child in path:
                    print("!!!环：{}".format(path + [child]))
                result[t, n, child, path[-1]] += transfer_traffic
                dfs(path + [child], t, n, transfer_traffic)

    for t in range(time):
        for s in range(n):
            accessed_svc = np.where(entry_services[t, s])[0]
            for svc in accessed_svc:
                result[t, s, svc, svc] = traffic[t, s, svc]
                dfs([svc], t, s, traffic[t, s, svc])

    return result


def compute_disseminated_workload(adj_matrix, entry_services, traffic):
    """
    Although we have constructed distinct workloads for each server,
    considering the evolving patterns of access over time,
    it is imperative to compute the aggregate traffic for each service at a given timestamp.
    This entails calculating the traffic originating from upstream services
    in addition to the requests directly received by the service itself,
    thereby providing a comprehensive measure of the total traffic received by the service
    at that particular time point.

    :param adj_matrix:
    :param entry_services:
    :param traffic:
    :return:
    """
    t, n, m = entry_services.shape
    t1, n, m = traffic.shape
    time = min(t, t1)
    result = np.zeros((time, n, adj_matrix.shape[0]))
    intersect_col = min(traffic.shape[-1], adj_matrix.shape[-1])
    result[..., :intersect_col] = traffic[..., :intersect_col]

    def dfs(root, t, n, transfer_traffic):
        # The dependency graph is guaranteed to be loop-free
        for child in range(adj_matrix.shape[0]):
            if adj_matrix[root, child] == 1:
                result[t, n, child] += transfer_traffic
                dfs(child, t, n, transfer_traffic)

    for t in range(time):
        for s in range(n):
            accessed_svc = np.where(entry_services[t, s])[0]
            for svc in accessed_svc:
                dfs(svc, t, s, traffic[t, s, svc])

    return result


# Example usage
training_duration_seconds = 1 * 24 * 60 * 60 + 2 * 60 * 60 + 30 * 60 + 15
formatted_duration = format_training_duration(training_duration_seconds)
print("Training duration:", formatted_duration)


def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
