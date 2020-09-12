import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video.resnet import r2plus1d_18




class R2Plus1D(nn.Module):

    def __init__(self,
                resize_input=True,
                normalize_input=True,
                requires_grad=False,
                use_fid_inception=True):
        super(R2Plus1D, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        video_resnet = r2plus1d_18(pretrained=True, progress=True)
        block = [
            video_resnet.stem,
            video_resnet.layer1,
            video_resnet.layer2,
            video_resnet.layer3,
            video_resnet.layer4,
            video_resnet.avgpool,
        ]
        self.blocks = nn.Sequential(*block)

        for param in self.parameters():
            param.requires_grad = requires_grad



    def forward(self, inputs):
        x = inputs
        if self.resize_input:
            batch, time, channel, w, h = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(batch*time, channel, w, h)
            x = F.interpolate(x,
                              size=(112, 112),
                              mode='bilinear',
                              align_corners=False)
            x = x.reshape(batch, time, channel, 112, 112).permute(0, 2, 1, 3, 4)
        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        output = self.blocks(inputs)
        return output


if __name__ == "__main__":
    model = R2Plus1D()
    T = 6
    inputs = torch.randn((32, 3, T, 64, 64))
    activation = model(inputs)
    print(activation.shape)