import torch, torchvision
import argparse, json


if __name__ == '__main__':
    parser = argparse.ArgumentParser("与联邦学习训练对比训练")
    parser.add_argument('--c', '--conf', dest='conf', default='Fedrated_Learning/utils/conf.json')
    parser.add_argument('--path', default="Fedrated_Learning/data/CIFAR10_Datasets/cifar-10-python.tar.gz", help="数据集路径")
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    # 1. 加载数据集和测试集
    dir = args.path

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4919, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4919, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))
    ])

    # 先读入数据集
    train_dataset = torchvision.datasets.CIFAR10(root=dir, train=True, transform=transform_train, download=True)
    eval_dataset = torchvision.datasets.CIFAR10(root=dir, train=False, transform=transform_test, download=True)

    # 给数据集进行批次划分
    train_batches = torch.utils.data.DataLoader(train_dataset, batch_size=conf['batch_size'],shuffle=True)
    eval_batches = torch.utils.data.DataLoader(eval_dataset, batch_size=conf['batch_size'])

    # 2. 加载模型resnet18
    model = torchvision.models.resnet18(pretrained=True)

    # 改变全连接层，及输出维度
    num_in_fc = model.fc.in_features  # in_features 表示该全连接层的输入特征数量（即上一层网络输出的特征维度）
                                    # 对于预训练的 ResNet18，这一层的输入特征数固定为 512，所以num_in_fc会得到 512
    model.fc = torch.nn.Linear(num_in_fc, 10)

    model = model.to("cuda")


    # 3. 开始训练
    optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'], momentum=conf['momentum']) # 设置优化器

    for epoch in range(conf['local_epochs'] * conf['global_epochs']):
        model.train()
        print("========= Epoch %d ===========" %(epoch+1))
        for batch_id, batch in enumerate(train_batches):
            data, target = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # 4. 测试
        model.eval()
        total_loss = 0
        acc = 0
        dataset_size = 0
        correct = 0
        for batch_id, batch in enumerate(eval_batches):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(1) # 将10维输出转换为单个类别索引,每张图片对应一个预测类别
            correct += pred.eq(target).sum().item()
        print("Epoch %d done    Accuracy: %f,      Loss: %f"
              %(epoch+1,(float(correct) / float(dataset_size)) * 100, float(total_loss) / float(dataset_size)))










