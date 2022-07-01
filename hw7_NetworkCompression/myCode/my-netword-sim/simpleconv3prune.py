import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models.simpleconv3 import simpleconv3

# 剪枝配置参数
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='checkpoints/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the model')
parser.add_argument('--save', default='pruned/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 结果文件夹
resultdir = args.save + str(args.percent)
if not os.path.exists(resultdir):
    os.makedirs(resultdir)

# 定义并加载未剪枝的模型
model = simpleconv3(4)
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state'])
if args.cuda:
    model.cuda()
print(model)

# 记录每个BatchNormal 层的参数个数
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        # m.weight.data.shape = channels x 1
        total += m.weight.data.shape[0]

# 记录所有BN 层上每一个参数的绝对值
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m_size = m.weight.data.shape[0]
        bn[index:(index + m_size)] = m.weight.data.abs().clone()
        index += m_size

# 对所有BN层上所有参数进行排序, 根据裁剪比例获取截断值(thre)
# 从而获取要丢弃的子集
y, i = torch.sort(bn)  # 会按值的顺序排列
thre_index = int(total * args.percent)  # 按比例获取index 值
thre = y[thre_index]  # 获取到剪枝的截断值，小于该值的weight 会被剪枝
print('prun th=' + str(thre))

# 以下开始用掩码矩阵模拟剪枝后的效果
# 掩码矩阵: 每个位置上的值，只有0或者1,
# 当为0时，表示当前位置上的参数会被剪枝,因为此时是 0*w*x =0
# 当为1时，表示当前位置上的参数不会被剪枝,因为此时是 1*w*x =w*x
pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float()
        # 以下一行代码计算从开始到当前层，所有被剪枝的参数数量
        # mask.shape[0] 是当前BN层的weight 参数总个数
        # torch.sum(mask) 是mask矩阵所有值为1的累加，即相当于是1的个数，即保留的参数个数
        # 两者相减，得到当前层被剪枝的个数
        pruned = pruned + (mask.shape[0] - torch.sum(mask))

        # 以下两行代码，通过w/bias 与mask 矩阵相乘，得到剪枝后的w矩阵
        m.weight.data.mul_(mask)  # 根据掩膜调整缩放系数
        m.bias.data.mul(mask)  # 根据掩膜调整偏置系数

        cfg.append(int(torch.sum(mask)))  # 获得当前层保留的通道数
        cfg_mask.append(mask.clone())  # 保存当前层的掩膜矩阵
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

# 计算出整体的剪枝率
pruned_ratio = pruned / total
print('Pre-processing Successful!')

# 数据
data_dir = './data'
image_size = 60
crop_size = 48
nclass = 4
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Scale(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4) for x in ['train', 'val']}
train_loader = dataloaders['train']
test_loader = dataloaders['val']


# 对预剪枝后的模型计算准确率
def test(model):
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


# 用mask  掩码矩阵
# 模拟剪枝后的效果，查看剪枝后的模型准确率
acc = test(model)

# 真正剪枝
print(cfg)
newmodel = simpleconv3(4)
if args.cuda:
    newmodel.cuda()

# 计算参数量
num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(resultdir, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n" + str(cfg) + "\n")
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
    fp.write("threshold:" + str(thre))
    fp.write("Test accuracy: \n" + str(acc))

lay_id_in_cfg = 0
start_mask = torch.ones(3)  # 输入层的通道数
end_mask = cfg_mask[lay_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        # 第二步走到这里，BN层参数 (output_channel,0)  ，各个维度的参数 剪枝
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        lay_id_in_cfg += 1
        start_mask = end_mask.clone()
        if lay_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[lay_id_in_cfg]
    # 其实首先走到这里，卷积层参数，对各个维度进行剪枝
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # m0.weight.data.shape: output_channel x input_channel x kernel_weight x kernel_height
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # 输入通道掩膜
        w1 = w1[idx1.tolist(), :, :, :].clone()  # 输出通道掩膜
        m1.weight.data = w1.clone()

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0,(1,))
        m1.weight.data = m0.weight.data[:,idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg,
            'state_dict': newmodel.state_dict()},
           os.path.join(resultdir, 'pruned.pth.tar'))
print(newmodel)
model = newmodel
test(model)
