# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class MobileFaceNetLite(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(MobileFaceNetLite, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MobileFaceNetLite::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[stem]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[stem]/ReLU[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[0]/Conv2d[0]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[0]/ReLU[2]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[0]/Conv2d[3]/input.15
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[0]/ReLU[5]/input.19
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=64, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[1]/Conv2d[0]/input.21
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[1]/ReLU[2]/input.25
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[1]/Conv2d[3]/input.27
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[1]/ReLU[5]/input.31
        self.module_11 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[2]/Conv2d[0]/input.33
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[2]/ReLU[2]/input.37
        self.module_13 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[2]/Conv2d[3]/input.39
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[2]/ReLU[5]/input.43
        self.module_15 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=128, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[3]/Conv2d[0]/input.45
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[3]/ReLU[2]/input.49
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[3]/Conv2d[3]/input.51
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[3]/ReLU[5]/input.55
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=256, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[4]/Conv2d[0]/input.57
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[4]/ReLU[2]/input.61
        self.module_21 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[4]/Conv2d[3]/input.63
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[4]/ReLU[5]/input.67
        self.module_23 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=256, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[5]/Conv2d[0]/input.69
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[5]/ReLU[2]/input.73
        self.module_25 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[5]/Conv2d[3]/input.75
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[5]/ReLU[5]/input.79
        self.module_27 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=512, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[6]/Conv2d[0]/input.81
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[6]/ReLU[2]/input.85
        self.module_29 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[6]/Conv2d[3]/input.87
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[6]/ReLU[5]/input.91
        self.module_31 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=512, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[7]/Conv2d[0]/input.93
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[7]/ReLU[2]/input.97
        self.module_33 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[7]/Conv2d[3]/input.99
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[blocks]/Sequential[7]/ReLU[5]/input.103
        self.module_35 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[head_conv]/Conv2d[0]/input.105
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #MobileFaceNetLite::MobileFaceNetLite/Sequential[head_conv]/ReLU[2]/input
        self.module_37 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileFaceNetLite::MobileFaceNetLite/AdaptiveAvgPool2d[gap]/2751
        self.module_38 = py_nndct.nn.Module('nndct_flatten') #MobileFaceNetLite::MobileFaceNetLite/Flatten[flatten]/2754
        self.module_39 = py_nndct.nn.Linear(in_features=512, out_features=512, bias=True) #MobileFaceNetLite::MobileFaceNetLite/Linear[linear]/2757

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(output_module_0)
        output_module_0 = self.module_17(output_module_0)
        output_module_0 = self.module_18(output_module_0)
        output_module_0 = self.module_19(output_module_0)
        output_module_0 = self.module_20(output_module_0)
        output_module_0 = self.module_21(output_module_0)
        output_module_0 = self.module_22(output_module_0)
        output_module_0 = self.module_23(output_module_0)
        output_module_0 = self.module_24(output_module_0)
        output_module_0 = self.module_25(output_module_0)
        output_module_0 = self.module_26(output_module_0)
        output_module_0 = self.module_27(output_module_0)
        output_module_0 = self.module_28(output_module_0)
        output_module_0 = self.module_29(output_module_0)
        output_module_0 = self.module_30(output_module_0)
        output_module_0 = self.module_31(output_module_0)
        output_module_0 = self.module_32(output_module_0)
        output_module_0 = self.module_33(output_module_0)
        output_module_0 = self.module_34(output_module_0)
        output_module_0 = self.module_35(output_module_0)
        output_module_0 = self.module_36(output_module_0)
        output_module_0 = self.module_37(output_module_0)
        output_module_0 = self.module_38(input=output_module_0, start_dim=1, end_dim=-1)
        output_module_0 = self.module_39(output_module_0)
        return output_module_0
