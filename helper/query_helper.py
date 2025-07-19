import datetime
import os
import time
import urllib.parse

import numpy as np
import requests


class QueryHelper:
    DEBUG_FILE_PREFIX = os.sep.join([os.path.abspath(os.curdir), "debug"])
    RESPONSE_TIME = "response_time"
    CPU_UTILIZATION = "cpu_utilization"
    MEM_UTILIZATION = "mem_utilization"
    REQ_RATE = "req_rate"
    query_statement = {}
    PROMETHEUS_HOST = "xxxxx"
    PROMETHEUS_API_IP = PROMETHEUS_HOST + "api/v1/query?time={}&query={}"
    # 这里注意，因为要用到format函数，所以不是代表占位的'{'要double
    query_statement[CPU_UTILIZATION] = "sum(sum(rate(container_cpu_usage_seconds_total{" \
                                       "namespace='{0}',container='{1}'}[30s])) by (pod_name, namespace))" \
                                       " / " \
                                       "sum(container_spec_cpu_quota{namespace='{0}',container='{1}'} / 100000)"
    query_statement[MEM_UTILIZATION] = "sum(container_memory_usage_bytes{{container_name='{0}'}})" \
                                       " /" \
                                       " sum(container_spec_memory_limit_bytes{{container_name={0}}})"
    query_statement[RESPONSE_TIME] = "sum(rate(istio_request_duration_milliseconds_sum{{" \
                                     "reporter='destination',destination_workload_namespace={0}," \
                                     "destination_workload={1}}}[{2}]))" \
                                     "/" \
                                     "sum(rate(istio_request_duration_milliseconds_count{{" \
                                     "reporter='destination',destination_workload_namespace={0}," \
                                     "destination_workload={1}}}[{2}]))"
    query_statement[REQ_RATE] = "sum(rate(istio_requests_total{destination_workload_namespace='{0}'," \
                                "reporter='destination',destination_workload='{1}'}[30s]))"

    # query_statement[RESPONSE_TIME] = "rate(istio_request_duration_milliseconds_sum{{" \
    #                                  "reporter='destination',destination_workload_namespace={0}," \
    #                                  "destination_workload={1}}}[{2}])" \
    #                                  "/" \
    #                                  "rate(istio_request_duration_milliseconds_count{{" \
    #                                  "reporter='destination',destination_workload_namespace={0}," \
    #                                  "destination_workload={1}}}[{2}])"

    def __init__(self):
        QueryHelper.__init_debug_path()

    @staticmethod
    def __init_debug_path():
        if not os.path.exists(QueryHelper.DEBUG_FILE_PREFIX):
            os.mkdir(QueryHelper.DEBUG_FILE_PREFIX, 777)

    @staticmethod
    def get_duration_per_request(namespace, destination_workload, interval, query_start_time, query_last_time=0):
        """
        获取自[time-interval,time]区间内，namespace下destination_workload收到的每条请求平均耗时
        该函数会连续查询

        :param query_last_time:
        :param namespace:
        :param destination_workload:
        :param interval:
        :param query_start_time:
        :return: 每条请求平均耗时（单位：s）
        """
        response_time_seq = []
        query_start_time = time.mktime(time.strptime(query_start_time, '%Y-%m-%d %H:%M:%S'))
        cur_time = query_start_time - 1
        for _ in range(query_last_time):
            cur_time += 1
            res_time = QueryHelper.__query_restime(cur_time, namespace, destination_workload, interval)
            while np.isnan(res_time):
                time.sleep(5)
                res_time = QueryHelper.__query_restime(cur_time, namespace, destination_workload, interval)
            response_time_seq.append(res_time)
        print(response_time_seq)
        return np.mean(response_time_seq)

    @staticmethod
    def __query_restime(cur_time, namespace, destination_workload, interval):
        response_time = np.nan
        query_url = QueryHelper.PROMETHEUS_API_IP.format(cur_time, urllib.parse.quote_plus(
            QueryHelper.query_statement[QueryHelper.RESPONSE_TIME].format(namespace, destination_workload,
                                                                          interval)))
        request_json = requests.get(query_url).json()
        with open(os.sep.join([QueryHelper.DEBUG_FILE_PREFIX, "response_time.json"]), "w") as out_file:
            print(request_json, file=out_file)
            reponse_result = request_json['data']['result']
            if len(reponse_result) > 0:
                reponse_result = reponse_result[0]['value'][-1]
                if reponse_result != 'NaN':
                    response_time = round(eval(reponse_result) / 1000, 3)
        return response_time


if __name__ == '__main__':
    QueryHelper.get_duration_per_request("'tongshen'", "~'product.*'", "60s", "2022-3-22 01:04:12", 600)
