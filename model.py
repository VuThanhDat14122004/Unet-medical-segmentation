import torch
import torch.nn as nn

class Block_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        # If inplace=True, change the value of input tensor. If inplace=False, a new tensor will be created, the original input tensor remains unchanged.
        self.activation1 = nn.ReLU(inplace=False)
        self.activation2 = nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        # x = self.pool(x)
        return x

class encode_decode_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encode_decode_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU(inplace=False)
    def forward(self, x):
        return self.activation(self.conv1(x))

class Block_up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block_up_sample, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=2, stride=2)
    def forward(self, x):
        return self.up_sample(x)

# def crop_center(batch_en_feat, batch_dec_feat):
#     b1, c1, h1, w1 = batch_en_feat.size()
#     b2, c2, h2, w2 = batch_dec_feat.size()
#     h_s = (h1 - h2)//2
#     w_s = (w1 - w2)//2
#     return batch_en_feat[:,:,h_s: (h2+h_s), w_s: (w2 + w_s)]

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()
        # encode
        self.en_block1 = Block_conv(in_channels=in_channels, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en_block2 = Block_conv(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en_block3 = Block_conv(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en_block4 = Block_conv(in_channels=256, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en_block5 = encode_decode_block(in_channels=512, out_channels=1024)
        # decode
        self.de_block5 = encode_decode_block(in_channels=1024, out_channels=1024)
        self.up_sample4 = Block_up_sample(in_channels=1024, out_channels=512)
        self.up_conv4 = Block_conv(in_channels=1024, out_channels=512)

        self.up_sample3 = Block_up_sample(in_channels=512, out_channels=256)
        self.up_conv3 = Block_conv(in_channels=512, out_channels=256)

        self.up_sample2 = Block_up_sample(in_channels=256, out_channels=128)
        self.up_conv2 = Block_conv(in_channels=256, out_channels=128)

        self.up_sample1 = Block_up_sample(in_channels=128, out_channels=64)
        self.up_conv1 = Block_conv(in_channels=128, out_channels=64)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels,
                                    kernel_size=1, stride=1, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.en_block1(x)

        e2 = self.pool1(e1)
        e2 = self.en_block2(e2)

        e3 = self.pool2(e2)
        e3 = self.en_block3(e3)

        e4 = self.pool3(e3)
        e4 = self.en_block4(e4)

        e5 = self.pool4(e4)
        e5 = self.en_block5(e5)

        d5 = self.de_block5(e5)
        
        d4 = self.up_sample4(d5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up_sample3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up_sample2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up_sample1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.up_conv1(d1)

        res = self.final_conv(d1)
        res = self.activation(res)
        return res