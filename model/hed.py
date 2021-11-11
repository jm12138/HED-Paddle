import paddle
import paddle.nn as nn


class HEDBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, num_convs, with_pool=True):
        super().__init__()
        # VGG Block
        if with_pool:
            pool = nn.MaxPool2D(kernel_size=2, stride=2)
            self.add_sublayer('pool', pool)

        conv1 = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        relu = nn.ReLU()

        self.add_sublayer('conv1', conv1)
        self.add_sublayer('relu1', relu)

        for _ in range(num_convs-1):
            conv = nn.Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.add_sublayer(f'conv{_+2}', conv)
            self.add_sublayer(f'relu{_+2}', relu)

        self.layer_names = [name for name in self._sub_layers.keys()]

        # Socre Layer
        self.score = nn.Conv2D(
            in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        for name in self.layer_names:
            input = self._sub_layers[name](input)
        return input, self.score(input)


class HED(nn.Layer):
    def __init__(self,
                 channels=[3, 64, 128, 256, 512, 512],
                 nums_convs=[2, 2, 3, 3, 3],
                 with_pools=[False, True, True, True, True]):
        super().__init__()
        '''
        HED model implementation in Paddle.

        Fix the padding parameter and use simple Bilinear Upsampling.
        '''
        assert (len(channels) - 1) == len(nums_convs), '(len(channels) -1) != len(nums_convs).'

        # HED Blocks
        for index, num_convs in enumerate(nums_convs):
            block = HEDBlock(in_channels=channels[index], out_channels=channels[index+1], num_convs=num_convs, with_pool=with_pools[index])
            self.add_sublayer(f'block{index+1}', block)

        self.layer_names = [name for name in self._sub_layers.keys()]

        # Output Layers
        self.out = nn.Conv2D(in_channels=len(nums_convs), out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        h, w = input.shape[2:]
        scores = []
        for index, name in enumerate(self.layer_names):
            input, score = self._sub_layers[name](input)
            if index > 0:
                score = nn.functional.upsample(score, size=[h, w], mode='bilinear')
            scores.append(score)

        output = self.out(paddle.concat(scores, 1))
        return self.sigmoid(output)


if __name__ == '__main__':
    model = HED()
    out = model(paddle.randn((1, 3, 256, 256)))
    print(out.shape)
