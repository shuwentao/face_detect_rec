# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class QATWrap(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(QATWrap, self).__init__()
        self.module_0 = py_nndct.nn.Input() #QATWrap::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/Conv2dNormActivation[features]/Conv2dNormActivation[0]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/Conv2dNormActivation[features]/Conv2dNormActivation[0]/Hardswish[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=16, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/Conv2dNormActivation[0]/ReLU[2]/input.13
        self.module_5 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/AdaptiveAvgPool2d[avgpool]/input.15
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/Conv2d[fc1]/input.17
        self.module_7 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/ReLU[activation]/input.19
        self.module_8 = py_nndct.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/Conv2d[fc2]/input.21
        self.module_9 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/Hardsigmoid[scale_activation]/13274
        self.module_10 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/SqueezeExcitation[1]/input.23
        self.module_11 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[1]/Sequential[block]/Conv2dNormActivation[2]/Conv2d[0]/input.25
        self.module_12 = py_nndct.nn.Conv2d(in_channels=16, out_channels=72, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[2]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.29
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[2]/Sequential[block]/Conv2dNormActivation[0]/ReLU[2]/input.33
        self.module_14 = py_nndct.nn.Conv2d(in_channels=72, out_channels=72, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=72, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[2]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.35
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[2]/Sequential[block]/Conv2dNormActivation[1]/ReLU[2]/input.39
        self.module_16 = py_nndct.nn.Conv2d(in_channels=72, out_channels=24, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[2]/Sequential[block]/Conv2dNormActivation[2]/Conv2d[0]/input.41
        self.module_17 = py_nndct.nn.Conv2d(in_channels=24, out_channels=88, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.45
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/Sequential[block]/Conv2dNormActivation[0]/ReLU[2]/input.49
        self.module_19 = py_nndct.nn.Conv2d(in_channels=88, out_channels=88, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=88, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.51
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/Sequential[block]/Conv2dNormActivation[1]/ReLU[2]/input.55
        self.module_21 = py_nndct.nn.Conv2d(in_channels=88, out_channels=24, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/Sequential[block]/Conv2dNormActivation[2]/Conv2d[0]/input.57
        self.module_22 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[3]/input.59
        self.module_23 = py_nndct.nn.Conv2d(in_channels=24, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.61
        self.module_24 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.65
        self.module_25 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=96, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.67
        self.module_26 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.71
        self.module_27 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.73
        self.module_28 = py_nndct.nn.Conv2d(in_channels=96, out_channels=24, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.75
        self.module_29 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.77
        self.module_30 = py_nndct.nn.Conv2d(in_channels=24, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.79
        self.module_31 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/13616
        self.module_32 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/SqueezeExcitation[2]/input.81
        self.module_33 = py_nndct.nn.Conv2d(in_channels=96, out_channels=40, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[4]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.83
        self.module_34 = py_nndct.nn.Conv2d(in_channels=40, out_channels=240, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.87
        self.module_35 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.91
        self.module_36 = py_nndct.nn.Conv2d(in_channels=240, out_channels=240, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=240, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.93
        self.module_37 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.97
        self.module_38 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.99
        self.module_39 = py_nndct.nn.Conv2d(in_channels=240, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.101
        self.module_40 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.103
        self.module_41 = py_nndct.nn.Conv2d(in_channels=64, out_channels=240, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.105
        self.module_42 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/13768
        self.module_43 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/SqueezeExcitation[2]/input.107
        self.module_44 = py_nndct.nn.Conv2d(in_channels=240, out_channels=40, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.109
        self.module_45 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[5]/input.111
        self.module_46 = py_nndct.nn.Conv2d(in_channels=40, out_channels=240, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.113
        self.module_47 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.117
        self.module_48 = py_nndct.nn.Conv2d(in_channels=240, out_channels=240, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=240, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.119
        self.module_49 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.123
        self.module_50 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.125
        self.module_51 = py_nndct.nn.Conv2d(in_channels=240, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.127
        self.module_52 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.129
        self.module_53 = py_nndct.nn.Conv2d(in_channels=64, out_channels=240, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.131
        self.module_54 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/13922
        self.module_55 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/SqueezeExcitation[2]/input.133
        self.module_56 = py_nndct.nn.Conv2d(in_channels=240, out_channels=40, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.135
        self.module_57 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[6]/input.137
        self.module_58 = py_nndct.nn.Conv2d(in_channels=40, out_channels=120, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.139
        self.module_59 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.143
        self.module_60 = py_nndct.nn.Conv2d(in_channels=120, out_channels=120, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=120, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.145
        self.module_61 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.149
        self.module_62 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.151
        self.module_63 = py_nndct.nn.Conv2d(in_channels=120, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.153
        self.module_64 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.155
        self.module_65 = py_nndct.nn.Conv2d(in_channels=32, out_channels=120, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.157
        self.module_66 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/14076
        self.module_67 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/SqueezeExcitation[2]/input.159
        self.module_68 = py_nndct.nn.Conv2d(in_channels=120, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[7]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.161
        self.module_69 = py_nndct.nn.Conv2d(in_channels=48, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.165
        self.module_70 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.169
        self.module_71 = py_nndct.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=144, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.171
        self.module_72 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.175
        self.module_73 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.177
        self.module_74 = py_nndct.nn.Conv2d(in_channels=144, out_channels=40, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.179
        self.module_75 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.181
        self.module_76 = py_nndct.nn.Conv2d(in_channels=40, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.183
        self.module_77 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/14228
        self.module_78 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/SqueezeExcitation[2]/input.185
        self.module_79 = py_nndct.nn.Conv2d(in_channels=144, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.187
        self.module_80 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[8]/input.189
        self.module_81 = py_nndct.nn.Conv2d(in_channels=48, out_channels=288, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.191
        self.module_82 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.195
        self.module_83 = py_nndct.nn.Conv2d(in_channels=288, out_channels=288, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=288, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.197
        self.module_84 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.201
        self.module_85 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.203
        self.module_86 = py_nndct.nn.Conv2d(in_channels=288, out_channels=72, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.205
        self.module_87 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.207
        self.module_88 = py_nndct.nn.Conv2d(in_channels=72, out_channels=288, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.209
        self.module_89 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/14382
        self.module_90 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/SqueezeExcitation[2]/input.211
        self.module_91 = py_nndct.nn.Conv2d(in_channels=288, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[9]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.213
        self.module_92 = py_nndct.nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.217
        self.module_93 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.221
        self.module_94 = py_nndct.nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=576, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.223
        self.module_95 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.227
        self.module_96 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.229
        self.module_97 = py_nndct.nn.Conv2d(in_channels=576, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.231
        self.module_98 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.233
        self.module_99 = py_nndct.nn.Conv2d(in_channels=144, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.235
        self.module_100 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/14536
        self.module_101 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/SqueezeExcitation[2]/input.237
        self.module_102 = py_nndct.nn.Conv2d(in_channels=576, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.239
        self.module_103 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[10]/input.241
        self.module_104 = py_nndct.nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/Conv2dNormActivation[0]/Conv2d[0]/input.243
        self.module_105 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/Conv2dNormActivation[0]/Hardswish[2]/input.247
        self.module_106 = py_nndct.nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=576, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/Conv2dNormActivation[1]/Conv2d[0]/input.249
        self.module_107 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/Conv2dNormActivation[1]/Hardswish[2]/input.253
        self.module_108 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/AdaptiveAvgPool2d[avgpool]/input.255
        self.module_109 = py_nndct.nn.Conv2d(in_channels=576, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc1]/input.257
        self.module_110 = py_nndct.nn.ReLU(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/ReLU[activation]/input.259
        self.module_111 = py_nndct.nn.Conv2d(in_channels=144, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/Conv2d[fc2]/input.261
        self.module_112 = py_nndct.nn.Hardsigmoid(inplace=False) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/Hardsigmoid[scale_activation]/14692
        self.module_113 = py_nndct.nn.Module('nndct_elemwise_mul') #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/SqueezeExcitation[2]/input.263
        self.module_114 = py_nndct.nn.Conv2d(in_channels=576, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/Sequential[block]/Conv2dNormActivation[3]/Conv2d[0]/input.265
        self.module_115 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/InvertedResidual[features]/InvertedResidual[11]/input.267
        self.module_116 = py_nndct.nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/Conv2dNormActivation[features]/Conv2dNormActivation[12]/Conv2d[0]/input.269
        self.module_117 = py_nndct.nn.Hardswish(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/TVBackbone[backbone]/Conv2dNormActivation[features]/Conv2dNormActivation[12]/Hardswish[2]/input.273
        self.module_118 = py_nndct.nn.Conv2d(in_channels=576, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[lat5]/p5
        self.module_119 = py_nndct.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[lat4]/14836
        self.module_120 = py_nndct.nn.Module('nndct_shape') #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14844
        self.module_121 = py_nndct.nn.Module('nndct_shape') #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14848
        self.module_122 = py_nndct.nn.Interpolate() #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14853
        self.module_123 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/p4
        self.module_124 = py_nndct.nn.Conv2d(in_channels=24, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[lat3]/14874
        self.module_125 = py_nndct.nn.Module('nndct_shape') #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14882
        self.module_126 = py_nndct.nn.Module('nndct_shape') #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14886
        self.module_127 = py_nndct.nn.Interpolate() #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/14891
        self.module_128 = py_nndct.nn.Add() #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/input.275
        self.module_129 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[smooth3]/input.277
        self.module_130 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/ReLU[relu]/input.283
        self.module_131 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[smooth4]/input.279
        self.module_132 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/ReLU[relu]/input.289
        self.module_133 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/Conv2d[smooth5]/input.281
        self.module_134 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/FPNLite[fpn]/ReLU[relu]/input.295
        self.module_135 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/Sequential[stem]/Conv2d[0]/input.285
        self.module_136 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/Sequential[stem]/ReLU[1]/input.287
        self.module_137 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/Conv2d[cls]/14996
        self.module_138 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/Conv2d[box]/15015
        self.module_139 = py_nndct.nn.Conv2d(in_channels=64, out_channels=10, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/Conv2d[lm]/15034
        self.module_140 = py_nndct.nn.Cat() #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[0]/15037
        self.module_141 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/Sequential[stem]/Conv2d[0]/input.291
        self.module_142 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/Sequential[stem]/ReLU[1]/input.293
        self.module_143 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/Conv2d[cls]/15076
        self.module_144 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/Conv2d[box]/15095
        self.module_145 = py_nndct.nn.Conv2d(in_channels=64, out_channels=10, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/Conv2d[lm]/15114
        self.module_146 = py_nndct.nn.Cat() #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[1]/15117
        self.module_147 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/Sequential[stem]/Conv2d[0]/input.297
        self.module_148 = py_nndct.nn.ReLU(inplace=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/Sequential[stem]/ReLU[1]/input
        self.module_149 = py_nndct.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/Conv2d[cls]/15156
        self.module_150 = py_nndct.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/Conv2d[box]/15175
        self.module_151 = py_nndct.nn.Conv2d(in_channels=64, out_channels=10, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/Conv2d[lm]/15194
        self.module_152 = py_nndct.nn.Cat() #QATWrap::QATWrap/FaceDetector[model]/DetHead[heads]/ModuleList[2]/15197

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_5 = self.module_7(output_module_5)
        output_module_5 = self.module_8(output_module_5)
        output_module_5 = self.module_9(output_module_5)
        output_module_5 = self.module_10(input=output_module_5, other=output_module_0)
        output_module_5 = self.module_11(output_module_5)
        output_module_5 = self.module_12(output_module_5)
        output_module_5 = self.module_13(output_module_5)
        output_module_5 = self.module_14(output_module_5)
        output_module_5 = self.module_15(output_module_5)
        output_module_5 = self.module_16(output_module_5)
        output_module_17 = self.module_17(output_module_5)
        output_module_17 = self.module_18(output_module_17)
        output_module_17 = self.module_19(output_module_17)
        output_module_17 = self.module_20(output_module_17)
        output_module_17 = self.module_21(output_module_17)
        output_module_17 = self.module_22(input=output_module_17, other=output_module_5, alpha=1)
        output_module_23 = self.module_23(output_module_17)
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_23 = self.module_26(output_module_23)
        output_module_27 = self.module_27(output_module_23)
        output_module_27 = self.module_28(output_module_27)
        output_module_27 = self.module_29(output_module_27)
        output_module_27 = self.module_30(output_module_27)
        output_module_27 = self.module_31(output_module_27)
        output_module_27 = self.module_32(input=output_module_27, other=output_module_23)
        output_module_27 = self.module_33(output_module_27)
        output_module_34 = self.module_34(output_module_27)
        output_module_34 = self.module_35(output_module_34)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_38 = self.module_38(output_module_34)
        output_module_38 = self.module_39(output_module_38)
        output_module_38 = self.module_40(output_module_38)
        output_module_38 = self.module_41(output_module_38)
        output_module_38 = self.module_42(output_module_38)
        output_module_38 = self.module_43(input=output_module_38, other=output_module_34)
        output_module_38 = self.module_44(output_module_38)
        output_module_38 = self.module_45(input=output_module_38, other=output_module_27, alpha=1)
        output_module_46 = self.module_46(output_module_38)
        output_module_46 = self.module_47(output_module_46)
        output_module_46 = self.module_48(output_module_46)
        output_module_46 = self.module_49(output_module_46)
        output_module_50 = self.module_50(output_module_46)
        output_module_50 = self.module_51(output_module_50)
        output_module_50 = self.module_52(output_module_50)
        output_module_50 = self.module_53(output_module_50)
        output_module_50 = self.module_54(output_module_50)
        output_module_50 = self.module_55(input=output_module_50, other=output_module_46)
        output_module_50 = self.module_56(output_module_50)
        output_module_50 = self.module_57(input=output_module_50, other=output_module_38, alpha=1)
        output_module_50 = self.module_58(output_module_50)
        output_module_50 = self.module_59(output_module_50)
        output_module_50 = self.module_60(output_module_50)
        output_module_50 = self.module_61(output_module_50)
        output_module_62 = self.module_62(output_module_50)
        output_module_62 = self.module_63(output_module_62)
        output_module_62 = self.module_64(output_module_62)
        output_module_62 = self.module_65(output_module_62)
        output_module_62 = self.module_66(output_module_62)
        output_module_62 = self.module_67(input=output_module_62, other=output_module_50)
        output_module_62 = self.module_68(output_module_62)
        output_module_69 = self.module_69(output_module_62)
        output_module_69 = self.module_70(output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_69 = self.module_72(output_module_69)
        output_module_73 = self.module_73(output_module_69)
        output_module_73 = self.module_74(output_module_73)
        output_module_73 = self.module_75(output_module_73)
        output_module_73 = self.module_76(output_module_73)
        output_module_73 = self.module_77(output_module_73)
        output_module_73 = self.module_78(input=output_module_73, other=output_module_69)
        output_module_73 = self.module_79(output_module_73)
        output_module_73 = self.module_80(input=output_module_73, other=output_module_62, alpha=1)
        output_module_81 = self.module_81(output_module_73)
        output_module_81 = self.module_82(output_module_81)
        output_module_81 = self.module_83(output_module_81)
        output_module_81 = self.module_84(output_module_81)
        output_module_85 = self.module_85(output_module_81)
        output_module_85 = self.module_86(output_module_85)
        output_module_85 = self.module_87(output_module_85)
        output_module_85 = self.module_88(output_module_85)
        output_module_85 = self.module_89(output_module_85)
        output_module_85 = self.module_90(input=output_module_85, other=output_module_81)
        output_module_85 = self.module_91(output_module_85)
        output_module_92 = self.module_92(output_module_85)
        output_module_92 = self.module_93(output_module_92)
        output_module_92 = self.module_94(output_module_92)
        output_module_92 = self.module_95(output_module_92)
        output_module_96 = self.module_96(output_module_92)
        output_module_96 = self.module_97(output_module_96)
        output_module_96 = self.module_98(output_module_96)
        output_module_96 = self.module_99(output_module_96)
        output_module_96 = self.module_100(output_module_96)
        output_module_96 = self.module_101(input=output_module_96, other=output_module_92)
        output_module_96 = self.module_102(output_module_96)
        output_module_96 = self.module_103(input=output_module_96, other=output_module_85, alpha=1)
        output_module_104 = self.module_104(output_module_96)
        output_module_104 = self.module_105(output_module_104)
        output_module_104 = self.module_106(output_module_104)
        output_module_104 = self.module_107(output_module_104)
        output_module_108 = self.module_108(output_module_104)
        output_module_108 = self.module_109(output_module_108)
        output_module_108 = self.module_110(output_module_108)
        output_module_108 = self.module_111(output_module_108)
        output_module_108 = self.module_112(output_module_108)
        output_module_108 = self.module_113(input=output_module_108, other=output_module_104)
        output_module_108 = self.module_114(output_module_108)
        output_module_108 = self.module_115(input=output_module_108, other=output_module_96, alpha=1)
        output_module_108 = self.module_116(output_module_108)
        output_module_108 = self.module_117(output_module_108)
        output_module_108 = self.module_118(output_module_108)
        output_module_119 = self.module_119(output_module_73)
        output_module_120 = self.module_120(input=output_module_73, dim=2)
        output_module_121 = self.module_121(input=output_module_73, dim=3)
        output_module_122 = self.module_122(input=output_module_108, size=[output_module_120,output_module_121], scale_factor=None, mode='nearest')
        output_module_119 = self.module_123(input=output_module_119, other=output_module_122, alpha=1)
        output_module_124 = self.module_124(output_module_17)
        output_module_125 = self.module_125(input=output_module_17, dim=2)
        output_module_126 = self.module_126(input=output_module_17, dim=3)
        output_module_127 = self.module_127(input=output_module_119, size=[output_module_125,output_module_126], scale_factor=None, mode='nearest')
        output_module_124 = self.module_128(input=output_module_124, other=output_module_127, alpha=1)
        output_module_124 = self.module_129(output_module_124)
        output_module_124 = self.module_130(output_module_124)
        output_module_131 = self.module_131(output_module_119)
        output_module_131 = self.module_132(output_module_131)
        output_module_133 = self.module_133(output_module_108)
        output_module_133 = self.module_134(output_module_133)
        output_module_124 = self.module_135(output_module_124)
        output_module_124 = self.module_136(output_module_124)
        output_module_137 = self.module_137(output_module_124)
        output_module_138 = self.module_138(output_module_124)
        output_module_139 = self.module_139(output_module_124)
        output_module_137 = self.module_140(dim=1, tensors=[output_module_137,output_module_138,output_module_139])
        output_module_131 = self.module_141(output_module_131)
        output_module_131 = self.module_142(output_module_131)
        output_module_143 = self.module_143(output_module_131)
        output_module_144 = self.module_144(output_module_131)
        output_module_145 = self.module_145(output_module_131)
        output_module_143 = self.module_146(dim=1, tensors=[output_module_143,output_module_144,output_module_145])
        output_module_133 = self.module_147(output_module_133)
        output_module_133 = self.module_148(output_module_133)
        output_module_149 = self.module_149(output_module_133)
        output_module_150 = self.module_150(output_module_133)
        output_module_151 = self.module_151(output_module_133)
        output_module_149 = self.module_152(dim=1, tensors=[output_module_149,output_module_150,output_module_151])
        return (output_module_137,output_module_143,output_module_149)
