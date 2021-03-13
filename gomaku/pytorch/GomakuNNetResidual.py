import torch
import torch.nn as nn
import torch.nn.functional as F
from ..GomakuGame import GomakuGame
from utils import dotdict

# args = dotdict({
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': torch.cuda.is_available(),
#     'num_channels': 512,
# })


class InputBlock(nn.Module):
    def __init__(self, board_size: int, args: dotdict):
        super(InputBlock, self).__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(args.num_channels)

    def forward(self, s):
        s = s.view(-1, 1, self.board_size, self.board_size)
        s = F.relu(self.bn(self.conv(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self, in_channels: int, action_size: int, board_size: int):
        super(OutBlock, self).__init__()
        self.board_size = board_size
        self.actions_size = action_size
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*self.board_size**2, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(32*self.board_size**2, self.actions_size)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * self.board_size**2)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 32*self.board_size**2)
        p = self.fc(p)
        p = self.logsoftmax(p)
        return p, v


class GomakuNet(nn.Module):
    def __init__(self, game: GomakuGame, args: dotdict):
        super(GomakuNet, self).__init__()
        self.game = game
        self.args = args
        self.board_size = self.game.getBoardSize()[0]
        self.action_size = self.game.getActionSize()
        self.input = InputBlock(self.board_size, self.args)
        for block in range(self.args.res_blocks):
            setattr(self, f"res_{block}", ResBlock(self.args.num_channels, self.args.num_channels))
        self.out = OutBlock(self.args.num_channels, self.action_size, self.board_size)

    def forward(self, s):
        s = self.input(s)
        for block in range(self.args.res_blocks):
            s = getattr(self, f"res_{block}")(s)
        s = self.out(s)
        return s

#
#
#
#
# class ConvBlock(nn.Module):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         self.action_size = 7
#         self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#
#     def forward(self, s):
#         s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
#         s = F.relu(self.bn1(self.conv1(s)))
#         return s
#
#
# class ResBlock(nn.Module):
#     def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = F.relu(self.bn1(out))
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = F.relu(out)
#         return out
#
#
# class OutBlock(nn.Module):
#     def __init__(self):
#         super(OutBlock, self).__init__()
#         self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
#         self.bn = nn.BatchNorm2d(3)
#         self.fc1 = nn.Linear(3 * 6 * 7, 32)
#         self.fc2 = nn.Linear(32, 1)
#
#         self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
#         self.bn1 = nn.BatchNorm2d(32)
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#         self.fc = nn.Linear(6 * 7 * 32, 7)
#
#     def forward(self, s):
#         v = F.relu(self.bn(self.conv(s)))  # value head
#         v = v.view(-1, 3 * 6 * 7)  # batch_size X channel X height X width
#         v = F.relu(self.fc1(v))
#         v = torch.tanh(self.fc2(v))
#
#         p = F.relu(self.bn1(self.conv1(s)))  # policy head
#         p = p.view(-1, 6 * 7 * 32)
#         p = self.fc(p)
#         p = self.logsoftmax(p).exp()
#         return p, v
#
#
# class ConnectNet(nn.Module):
#     def __init__(self):
#         super(ConnectNet, self).__init__()
#         self.conv = ConvBlock()
#         for block in range(19):
#             setattr(self, "res_%i" % block, ResBlock())
#         self.outblock = OutBlock()
#
#     def forward(self, s):
#         s = self.conv(s)
#         for block in range(19):
#             s = getattr(self, "res_%i" % block)(s)
#         s = self.outblock(s)
#         return s