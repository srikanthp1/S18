import torch
import torch.nn as nn
import torch.nn.functional as F

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encode_pool='maxpool'):
        super(ContractingBlock, self).__init__()

        self.encode_pool = encode_pool

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        if self.encode_pool == 'maxpool':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.encode_pool == 'stridedconv':
            self.maxpool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        skip = x  # store the output for the skip connection
        if self.encode_pool == 'maxpool':
            x = self.maxpool(x)
        elif self.encode_pool == 'stridedconv':
            x = F.relu(self.maxpool(x))
        
        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, decode_oppool='upsample'):
        super(ExpandingBlock, self).__init__()
        self.decode_oppool = decode_oppool
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if self.decode_oppool == 'transpose':
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
            
    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.decode_oppool == 'transpose':
            x = F.relu(self.upsample(x))
        if self.decode_oppool == 'upsample':
            F.upsample(x, scale_factor=2, mode='nearest')
        
        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, encode_pool='maxpool', decode_oppool='upsample'):
        super(UNet, self).__init__()
        
        self.contract1 = ContractingBlock(in_channels, 64, encode_pool)
        self.contract2 = ContractingBlock(64, 128, encode_pool)
        self.contract3 = ContractingBlock(128, 256, encode_pool)
        self.contract4 = ContractingBlock(256, 512, encode_pool)
        
        self.expand1 = ExpandingBlock(512, 256, decode_oppool)
        self.expand2 = ExpandingBlock(256, 128, decode_oppool)
        self.expand3 = ExpandingBlock(128, 64, decode_oppool)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, _ = self.contract4(x)
        
        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)