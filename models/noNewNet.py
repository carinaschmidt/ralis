# original nnUNet by FabianIsensee: https://github.com/MIC-DKFZ/nnUNet
# adapted by RobinBruegge: https://github.com/RobinBruegger/PartiallyReversibleUnet/blob/master/experiments/noNewNet.py

#from comet_ml import Experiment, ExistingExperiment
import sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
from utils import bratsUtils
import torch.nn.functional as F
import random
import torch.utils.model_zoo as model_zoo
import math
import utils.parser as parser

#from memory_profiler import profile

id = random.getrandbits(64)

#restore experiment
#VALIDATE_ALL = False
PREDICT = False
#RESTORE_ID = 395
#RESTORE_EPOCH = 350
#LOG_COMETML_EXISTING_EXPERIMENT = ""

#general settings
SAVE_CHECKPOINTS = False #set to true to create a checkpoint at every epoch
EXPERIMENT_TAGS = ["bugfreeFinalDrop"]
EXPERIMENT_NAME = "Nonreversible NO_NEW30"
EPOCHS = 1000
BATCH_SIZE = 1
VIRTUAL_BATCHSIZE = 1
VALIDATE_EVERY_K_EPOCHS = 1
INPLACE = True

#hyperparameters
CHANNELS = 30
INITIAL_LR = 1e-4
L2_REGULARIZER = 1e-5

#logging settings
LOG_EVERY_K_ITERATIONS = 50 #0 to disable logging
LOG_MEMORY_EVERY_K_ITERATIONS = False
LOG_MEMORY_EVERY_EPOCH = True
LOG_EPOCH_TIME = True
LOG_VALIDATION_TIME = True
LOG_HAUSDORFF_EVERY_K_EPOCHS = 0 #must be a multiple of VALIDATE_EVERY_K_EPOCHS
LOG_COMETML = False
LOG_PARAMCOUNT = True
LOG_LR_EVERY_EPOCH = True

#data and augmentation
TRAIN_ORIGINAL_CLASSES = False #train on original 5 classes
DATASET_WORKERS = 1
SOFT_AUGMENTATION = False #Soft augmetation directly works on the 3 classes. Hard augmentation augments on the 5 orignal labels, then takes the argmax
NN_AUGMENTATION = True #Has priority over soft/hard augmentation. Uses nearest-neighbor interpolation
DO_ROTATE = True
DO_SCALE = True
DO_FLIP = True
DO_ELASTIC_AUG = True
DO_INTENSITY_SHIFT = True
RANDOM_CROP = [128, 128, 128]

ROT_DEGREES = 20
SCALE_FACTOR = 1.1
SIGMA = 10
MAX_INTENSITY_SHIFT = 0.1

# if LOG_COMETML:
#     if not "LOG_COMETML_EXISTING_EXPERIMENT" in locals():
#         experiment = Experiment(api_key="", project_name="", workspace="")
#     else:
#         experiment = ExistingExperiment(api_key="", previous_experiment=LOG_COMETML_EXISTING_EXPERIMENT, project_name="", workspace="")
# else:
experiment = None

#network funcitons
if TRAIN_ORIGINAL_CLASSES:
    loss = bratsUtils.bratsDiceLossOriginal5
else:
    #loss = bratsUtils.bratsDiceLoss
    def loss(outputs, labels):
        return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)


class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
        super(EncoderModule, self).__init__()
        torch.cuda.empty_cache()
        groups = min(outChannels, CHANNELS)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(groups, outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)
    
    #@profile
    def forward(self, x):
        torch.cuda.empty_cache()
        if self.maxpool:
            x = F.max_pool3d(x, 2)
        doInplace = INPLACE and not self.hasDropout
        x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=doInplace)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
        super(DecoderModule, self).__init__()
        torch.cuda.empty_cache()
        groups = min(outChannels, CHANNELS)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(groups, inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, outChannels)
    
   # @profile
    def forward(self, x):
        torch.cuda.empty_cache()
        if self.firstConv:
            x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=INPLACE)
        x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=INPLACE)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class NoNewNet(nn.Module):
    def __init__(self):
        super(NoNewNet, self).__init__()

        #@carina use num_classes as channels
        print("in NoNewNet")
        torch.cuda.empty_cache()
        channels = CHANNELS
        #channels = CHANNELS
        self.levels = 5

        self.lastConv = nn.Conv3d(channels, 3, 1, bias=True)
        #nn.Conv3d()

        #create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(4, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), True, False))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), True, True))
        decoderModules.append(DecoderModule(channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)
    
   # @profile
    def forward(self, x):
        torch.cuda.empty_cache()
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()
        #print("x before last conv", x.shape)
        x = self.lastConv(x)
        #print("x before sigmoid", x.shape)
        x = torch.sigmoid(x)
        #print("x after all", x.shape)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 freezed=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if freezed:
            for i in self.bn1.parameters():
                i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if freezed:
            for i in self.bn2.parameters():
                i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if freezed:
            for i in self.bn3.parameters():
                i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.ModuleList([nn.Conv2d(inplanes,
                                                  planes * self.expansion,
                                                  kernel_size=1, stride=stride,
                                                  bias=False),
                                        nn.BatchNorm2d(
                                            planes * self.expansion)])
            if freezed:
                for i in downsample[1].parameters():
                    i.requires_grad = False
        self.downsample = downsample

    def forward(self, x):
        print("x.shape: ", x.shape)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample[0](residual)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000,
                 freezed=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if freezed:
            for i in self.bn1.parameters():
                i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], freezed=freezed)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       freezed=freezed)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       freezed=freezed)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       freezed=freezed)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, freezed=False):
        downsample = None

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, freezed=freezed))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, l in enumerate(self.layer1):
            x = l(x)
        for i, l in enumerate(self.layer2):
            x = l(x)
        for i, l in enumerate(self.layer3):
            x = l(x)
        for i, l in enumerate(self.layer4):
            x = l(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# net = NoNewNet()

#optimizer = optim.Adam(net.parameters(), lr=INITIAL_LR, weight_decay=L2_REGULARIZER)
#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [250, 400, 550], 0.2)

# @carina changed preTrained to True to use ImageNet
# def NoNewNet_(num_classes, pretrained=False,
#           freezed=False):
#     args = parser.get_arguments()
#     if "ImageNetBackbone" in args.exp_name:
#         pretrained = True
#     else:
#         pretrained = False
#     print("in NoNewNet: pretrained = ", pretrained)
#     model = NoNewNet(num_classes=num_classes,
#                 pretrained=pretrained, freezed=freezed)
#     return model