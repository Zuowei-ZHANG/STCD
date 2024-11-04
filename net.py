import torch.nn as nn
import torch.nn.functional as F
import torch


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# adaptive_instance_normalization / adaIN
def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      padding=0), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding=0), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=0), nn.ReLU())

        # decoder
        self.decoder = nn.Sequential(
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                      padding=0))

        self.mse_loss = nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.sl1_loss = nn.SmoothL1Loss()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CDCAEnet_diff_size(nn.Module):
    def __init__(self):
        super(CDCAEnet_diff_size, self).__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      padding=0), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding=0), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=0), nn.ReLU())
        '''
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.enc2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        '''
        enc_layers = list(self.enc.children())
        self.enc_1 = nn.Sequential(*enc_layers[:2])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[2:6])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[6:10])  # relu2_1 -> relu3_1
        for name in ['enc_1', 'enc_2', 'enc_3']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        # decoder
        self.dec = nn.Sequential(
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            # nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                      padding=0))
        '''
        self.dec1 = nn.ConvTranspose2d(in_channels=64,
                                       out_channels=32,
                                       kernel_size=3)
        self.dec2 = nn.ConvTranspose2d(in_channels=32,
                                       out_channels=3,
                                       kernel_size=3)
        '''
        self.mse_loss = nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.sl1_loss = nn.SmoothL1Loss()

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, beta=0.7):
        #x = F.relu(self.enc1(x))
        #style_feat = self.enc(style)
        style_feat = self.encode_with_intermediate(style)
        content_feat = self.enc(content)

        t = adain(content_feat, style_feat[-1])
        t = alpha * t + (1 - alpha) * content_feat
        #t.requires_grad_(False)
        g_t = self.dec(t)
        g_t_feats = self.enc(g_t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_loss(g_t_feats[-1],
                                t)  # + self.calc_loss(g_t, style)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feat[0])
        for i in range(1, 3):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feat[i])
        loss_ss = (1 - beta) * self.calc_loss(
            g_t, content) + beta * self.calc_loss(g_t, style)
        #loss_c = self.calc_loss(g_t, content)
        return g_t, loss_s + loss_c + loss_ss


class CDCAEnet(nn.Module):
    def __init__(self):
        super(CDCAEnet, self).__init__()
        # encoder
        self.enc = nn.Sequential(
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.ReLU())
        '''
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.enc2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        '''
        enc_layers = list(self.enc.children())
        self.enc_1 = nn.Sequential(*enc_layers[:2])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[2:4])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[4:6])  # relu2_1 -> relu3_1
        for name in ['enc_1', 'enc_2', 'enc_3']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               padding=1), nn.ReLU(),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3,
                      padding=1))
        '''
        self.dec1 = nn.ConvTranspose2d(in_channels=64,
                                       out_channels=32,
                                       kernel_size=3)
        self.dec2 = nn.ConvTranspose2d(in_channels=32,
                                       out_channels=3,
                                       kernel_size=3)
        '''
        self.mse_loss = nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.sl1_loss = nn.SmoothL1Loss()

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, beta=0.7):
        #x = F.relu(self.enc1(x))
        #style_feat = self.enc(style)
        style_feat = self.encode_with_intermediate(style)
        content_feat = self.enc(content)

        t = adain(content_feat, style_feat[-1])
        t = alpha * t + (1 - alpha) * content_feat
        #t.requires_grad_(False)
        g_t = self.dec(t)
        g_t_feats = self.enc(g_t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_loss(g_t_feats[-1],
                                t)  # + self.calc_loss(g_t, style)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feat[0])
        for i in range(1, 3):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feat[i])
        loss_ss = (1 - beta) * self.calc_loss(
            g_t, content) + beta * self.calc_loss(g_t, style)
        #loss_c = self.calc_loss(g_t, content)
        return g_t, loss_s + loss_c + loss_ss


def test():
    net = CDCAEnet()
    y = net(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
    print(y)


if __name__ == "__main__":
    test()