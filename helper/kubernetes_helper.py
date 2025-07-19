from __future__ import print_function

import logging
import os.path
import time
from pprint import pprint

import kubernetes
from kubernetes.client import ApiException
from logging import handlers
from helper import util

"""
新的k8s API主要是就是通过http请求动态修改配置文件然后上传
该类主要是一个kubernetes帮助类，实现了各类kubernetes的操作等等
"""

# root_dir = util.get_project_root()
# LOG_PATH = os.sep.join([root_dir, "log"])
# if not os.path.exists(LOG_PATH):
#     os.mkdir(LOG_PATH)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
# logger = logging.getLogger(__file__)
# handler = handlers.RotatingFileHandler(os.sep.join([LOG_PATH, os.path.basename(__file__).split(".")[0] + ".log"]),
#                                        encoding='utf-8',
#                                        maxBytes=104857600, backupCount=2)
# fomatter = logging.Formatter('%(asctime)s: %(message)s')
# handler.setFormatter(fomatter)
# logger.addHandler(handler)
logger = util.get_logger()


class KubernetesHelper:
    DEBUG_FILE_PREFIX = os.sep.join([os.path.abspath(os.curdir), "debug"])
    # 部署实例的命名空间
    KUBERNETES_NAMESPACE = "default"
    DEPLOYMENT_NAME = "productpage-v1"
    # kubernetes配置，主要是认证信息和主机域名
    configuration = kubernetes.client.Configuration()
    configuration.api_key_prefix['authorization'] = 'xxxx'
    # 这里通过命令
    # kubectl config view -o jsonpath='{"Cluster name\tServer\n"}{range .clusters[*]}{.name}{"\t"}{.cluster.server}{"\n"}{end}'
    # 获取，端口一般不变
    configuration.host = "xxxx"
    configuration.verify_ssl = False

    # 修改pod数的命令，参数：副本数 deployment
    PATCH_REPLICAS = "kubectl scale --replicas={} deployment/{}"

    def __init__(self) -> None:
        self.apps_api, self.core_api = KubernetesHelper.__init_api_instance__()
        KubernetesHelper.__init_debug_path()

    @staticmethod
    def __init_api_instance__():
        with kubernetes.client.ApiClient(KubernetesHelper.configuration) as api_client:
            apps_api = kubernetes.client.AppsV1Api(api_client)
            core_api = kubernetes.client.CoreV1Api(api_client)
        return apps_api, core_api

    @staticmethod
    def __init_debug_path():
        if not os.path.exists(KubernetesHelper.DEBUG_FILE_PREFIX):
            os.mkdir(KubernetesHelper.DEBUG_FILE_PREFIX, 777)

    @staticmethod
    def patch_replicas_cmd(replicas, deployment):
        os.system(KubernetesHelper.PATCH_REPLICAS.format(replicas, deployment))

    def create_deployment(self, namespace):
        """
        创建一个k8s的deployment
        :param namespace: k8s的命名空间
        """
        body = kubernetes.client.V1Deployment()
        pretty = 'pretty_example'  # str | If 'true', then the output is pretty printed. (optional)
        dry_run = 'dry_run_example'  # str | When present, indicates that modifications should not be persisted. An invalid or unrecognized dryRun directive will result in an error response and no further processing of the request. Valid values are: - All: all dry run stages will be processed (optional)
        field_manager = 'field_manager_example'  # str | fieldManager is a name associated with the actor or entity that is making these changes. The value must be less than or 128 characters long, and only contain printable characters, as defined by https://golang.org/pkg/unicode/#IsPrint. (optional)
        field_validation = 'field_validation_example'
        try:
            api_response = self.apps_api.create_namespaced_deployment(namespace, body, pretty=pretty,
                                                                      dry_run=dry_run,
                                                                      field_manager=field_manager,
                                                                      field_validation=field_validation)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling AppsV1Api->create_namespaced_deployment: %s\n" % e,
                  file=open("E:\Paper\云计算\项目\项目代码\多服务扩缩代码\error.md", "w"))

    def list_all_deployments(self) -> None:
        """
        列举所有命名空间下的deployments
        """
        with open(os.path.abspath(__file__) + os.sep + ".." + os.sep + "deployment.json", "w") as out:
            print(self.apps_api.list_deployment_for_all_namespaces(timeout_seconds=60),
                  file=out)

    def get_deployment(self, deployment_name, namespace):
        with open(os.sep.join([KubernetesHelper.DEBUG_FILE_PREFIX, "deployment.json"]), "w") as debug_file:
            deployment = self.apps_api.read_namespaced_deployment(deployment_name, namespace)
            print(deployment, file=debug_file)
        return deployment

    def patch_deployment_by_namespace(self, deployment_name, namespace, num_replicas):
        """
        修改kubernetes deployment配置文件
        :param deployment_name:
        :param namespace:
        :param num_replicas:
        """
        threshold = 5 * 60
        start = time.time()
        while True:
            try:
                deployment = self.get_deployment(deployment_name, namespace)
                deployment.spec.replicas = num_replicas
                self.apps_api.patch_namespaced_deployment(deployment_name, namespace, deployment)
            except ApiException:
                time.sleep(10)
            else:
                logger.info("{}修改成功!修改后：{}个pods".format(deployment_name, num_replicas))
                break
            finally:
                if time.time() - start > threshold:
                    logger.error("修改pod数超时失败！")
                    break

    def get_node(self, node_name):
        return self.core_api.read_node(node_name)

    def nominate_pod_to_node(self, deployment_name, namespace, node_name):
        """
        指定pod部署到给定的node

        :param deployment_name:
        :param namespace:
        :param node_name:
        """
        threshold = 5 * 60
        start = time.time()
        while True:
            try:
                node = self.get_node(node_name)
                node.metadata.labels = {"node_name": node_name}
                self.core_api.patch_node(node_name, node)
                deployment = self.get_deployment(deployment_name, namespace)
                deployment.spec.template.spec.node_selector = {"node_name": node_name}
                self.apps_api.patch_namespaced_deployment(deployment_name, namespace, deployment)
            except ApiException:
                time.sleep(10)
            else:
                logger.info("修改节点部署成功！")
                break
            finally:
                if time.time() - start > threshold:
                    break


if __name__ == "__main__":

    k8sObject = KubernetesHelper()
    for deploy in ["productpage-v1", "details-v1", "ratings-v1", "reviews-v1", "reviews-v2", "reviews-v3"]:
        k8sObject.patch_deployment_by_namespace(deploy,
                                                KubernetesHelper.KUBERNETES_NAMESPACE,
                                                1)
