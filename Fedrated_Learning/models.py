import torch
from torchvision import models

def get_model(name="resnet18", pretrained=True):
    if name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    else:
        model = models.resnet18(pretrained=pretrained)

    # 改变全连接层，及输出维度
    num_in_fc = model.fc.in_features # in_features 表示该全连接层的输入特征数量（即上一层网络输出的特征维度）
                                    # 对于预训练的 ResNet18，这一层的输入特征数固定为 512，所以num_in_fc会得到 512
    model.fc = torch.nn.Linear(num_in_fc, 10)

    if torch.cuda.is_available():
        model = model.cuda()
    return model
