from mkl_random.mklrand import shuffle
import models
import torch


class Client:
    def __init__(self, conf, train_dataset, id = -1):
        self.conf = conf     # 导入配置文件
        self.local_model = models.get_model(self.conf["model_name"])     # 导入本地模型（与全局模型相同）
        self.client_id = id    # 客户端数
        self.train_dataset = train_dataset   # 导入训练集
        all_range = list(range(len(self.train_dataset)))    # 对每一张图片进行创建索引
        data_len = int(len(self.train_dataset) / self.conf["no_models"])   # 每个客户端划分的集合大小
        train_indices = all_range[id * data_len: (id + 1) * data_len]    # 用用户ID做样本切片，每个客户端都有属于自己的样本，样本之间没有交集


        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.conf["batch_size"],
                                                            sampler=torch.utils.data.SubsetRandomSampler(train_indices))
        # SubsetRandomSampler：PyTorch 提供的采样器，作用是从完整数据集中只抽取 train_indices 列表中指定的索引对应的样本，且在每个 epoch 会随机打乱这些样本的顺序

    def local_train(self, model):   #用于客户端本地训练，其中model是服务器发放给客户端的全局模型
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())   #将全局模型的参数更新客户端本地模型参数

        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr = self.conf['lr'], momentum=self.conf["momentum"])   # 设置优化器

        self.local_model.train()   # 将本地模型切换到训练模式
        for epoch in range(self.conf["local_epochs"]):   # 本地模型开始训练
            for batch_id, batch in enumerate(self.train_dataloader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = self.local_model(data)   # 输出模型预测图片属于各类标签的概率
                loss = torch.nn.functional.cross_entropy(output, target)   # 计算预测各类标签值和真实标签值之间的交叉熵损失

                optimizer.zero_grad()   # 先清除上一次计算的梯度，防止梯度累加，导致难以收敛
                loss.backward()    # 反向计算梯度
                optimizer.step()    # 根据逆梯度方向更新参数，以减少损失值

            print("Epoch {} done ".format(epoch + 1))

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = data - model.state_dict()[name]    # 计算本地模型训练后参数的更新量
        return diff     # 返回参数更新量，以便后续上传给服务器做准备








