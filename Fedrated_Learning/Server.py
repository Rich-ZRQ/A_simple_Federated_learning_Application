import torch, models
from sympy.core.random import shuffle




class Server:
    def __init__(self, conf, eval_dataset):
        self.conf = conf    # 传入配置文件
        self.global_model = models.get_model(self.conf["model_name"])    # 传入全局模型
        self.eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                batch_size=self.conf["batch_size"],shuffle=True)   # 加载评估数据集

    def model_aggregate(self, weight_accumulator):     # 全局模型更新（聚合各本地训练后的模型参数然后利用算法更新）
        for name, data in self.global_model.state_dict().items():   # 遍历全局模型的每一层，包括模块名字及其参数
            update_this_layer = weight_accumulator[name] * self.conf["lambda"]  # FedAvg算法公式一部分
            if data.type() == update_this_layer.type(): data.add_(update_this_layer)   # FedAvg算法公式一部分
            else: data.add_(update_this_layer.to(torch.int64))

    def model_eval(self):   #评估模型性能
        self.global_model.eval()    #模型模式切换为评估模式

        total_loss = 0    # 累计样本的所有损失总和
        correct = 0     # 累计预测正确的样本数量
        dataset_size = 0   # 累计处理的总样本数量

        for batch_id, batch in enumerate(self.eval_loader):
            data,target = batch   # 分为数据集和对应标签
            dataset_size += data.size()[0]  # 在第0维度就是图片的张数
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

                output = self.global_model(data)    # 将数据放入更新后的全局模型，输出各类标签的预测值
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()   # 计算预测值和真实值之间的总交叉熵损失之和， 其中.item()是转化为python的浮点类型

                pred = output.argmax(1)    # 表示在第一维度上类别维度取最大值，返回一个元组（最大值，最大值索引）取最大值索引赋给pred
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()   #target.data.view_as(pred)确保真实标签的形状与预测结果pred一致
                #.cpu() 将张量从 GPU 转移到 CPU（若之前用了 GPU）
                # .sum() 计算当前批次正确的样本数（True 会被转换为 1，False 为 0）
                # .item() 将结果转换为 Python 数值并累加到 correct 中

        acc = 100.0 * (float(correct) / float(dataset_size))    # 计算准确率
        loss = total_loss / dataset_size     # 计算损失值
        return acc, loss












