import argparse, json

import torch

import datasets
from Server import *
from Client import *
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated learning')   # 创建参数解析器
    parser.add_argument("--c", "--conf", dest='conf')     # 加入参数 --c or --conf,dest中的作用是把调用该参树的名设置为conf，及后续是args.conf调用它
    args = parser.parse_args()

    with open(args.conf, 'r') as f:   # 读取文件
        conf = json.load(f)     # 读取文件中的json格式文件


    train_dataset, eval_dataset = datasets.get_dataset("./data/CIFAR10_Datasets/cifar-10-python.tar.gz", conf['type'])   # 获取训练集和测试集

    server = Server(conf, eval_dataset)

    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, train_dataset, c))    # 按照配置文件，设置客户端数量，并都加入clients列表


    for epoch in range(conf["global_epochs"]):    # 开始全局训练
        print("=========== Epoch %d ==============" %(epoch+1))
        candidates = random.sample(clients, conf['k'])    # 随机选取一定数量的客户端进行训练

        weight_accumulator = {}   # 存储客户端训练后模型与全局模型相比的更新总和
        for name, param in server.global_model.state_dict().items():
            # 对于全局变量的权重累加操作首先先置为0张量（大小与每层参数张量形状大小相同），表示目前全局模型的参数无需更改
            weight_accumulator[name] = torch.zeros_like(param)

        for c in candidates:
            diff = c.local_train(server.global_model)    # 对于每个客户端进行训练，并返回与全局模型参数之间的差异值（也就是本地模型训练后的参数更新变化）

            for name, param in server.global_model.state_dict().items():   # 累计各客户端权重变化
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)   # 经典算法 FedAvg
        acc, loss = server.model_eval()   # 全局模型在测试集上测试后的精度和损失值大小
        print("Epoch:%d, Accuracy:%f, Loss:%f" %(epoch+1, acc, loss))














