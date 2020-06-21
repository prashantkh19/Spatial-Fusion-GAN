
# Spatial-Fusion-GAN

## Network Architectures

```
GeometrySynthesizer(
  (stn_tpn): STN_TPN(
    (loc): Sequential(
      (0): Conv2d(6, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
      (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (8): ReLU(inplace=True)
    )
    (fc_loc): Sequential(
      (0): Linear(in_features=115200, out_features=512, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=512, out_features=76, bias=True)
      (3): Tanh()
    )
  )
  (stn_affine): STN_AFFINE(
    (loc): Sequential(
      (0): Conv2d(6, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (2): ReLU(inplace=True)
      (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): ReLU(inplace=True)
      (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
      (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (8): ReLU(inplace=True)
    )
    (fc_loc): Sequential(
      (0): Linear(in_features=115200, out_features=512, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=512, out_features=6, bias=True)
    )
  )
)

Generator(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
    (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (3): ReLU(inplace=True)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): ReLU(inplace=True)
    (10): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (11): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (12): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (13): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (14): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (15): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (16): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (17): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (18): ResidualBlock(
      (conv_block): Sequential(
        (0): ReflectionPad2d((1, 1, 1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): ReflectionPad2d((1, 1, 1, 1))
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (19): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (20): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (21): ReLU(inplace=True)
    (22): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (23): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (24): ReLU(inplace=True)
    (25): ReflectionPad2d((3, 3, 3, 3))
    (26): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (27): Tanh()
  )
  (filter): GuidedFilter()
)

Discriminator(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Dropout(p=0.1, inplace=False)
    (3): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (4): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (5): LeakyReLU(negative_slope=0.2, inplace=True)
    (6): Dropout(p=0.1, inplace=False)
    (7): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (9): LeakyReLU(negative_slope=0.2, inplace=True)
    (10): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  )
)

```
## Network Summary
### 1. Geometric Synthesizer

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 32, 254, 254]           1,760
             MaxPool2d-2         [-1, 32, 127, 127]               0
                  ReLU-3         [-1, 32, 127, 127]               0
                Conv2d-4         [-1, 64, 125, 125]          18,496
             MaxPool2d-5           [-1, 64, 62, 62]               0
                  ReLU-6           [-1, 64, 62, 62]               0
                Conv2d-7          [-1, 128, 60, 60]          73,856
             MaxPool2d-8          [-1, 128, 30, 30]               0
                  ReLU-9          [-1, 128, 30, 30]               0
               Linear-10                  [-1, 512]      58,982,912
                 ReLU-11                  [-1, 512]               0
               Linear-12                    [-1, 6]           3,078
           STN_AFFINE-13          [-1, 3, 256, 256]               0
               Conv2d-14         [-1, 32, 254, 254]           1,760
            MaxPool2d-15         [-1, 32, 127, 127]               0
                 ReLU-16         [-1, 32, 127, 127]               0
               Conv2d-17         [-1, 64, 125, 125]          18,496
            MaxPool2d-18           [-1, 64, 62, 62]               0
                 ReLU-19           [-1, 64, 62, 62]               0
               Conv2d-20          [-1, 128, 60, 60]          73,856
            MaxPool2d-21          [-1, 128, 30, 30]               0
                 ReLU-22          [-1, 128, 30, 30]               0
               Linear-23                  [-1, 512]      58,982,912
                 ReLU-24                  [-1, 512]               0
               Linear-25                   [-1, 76]          38,988
                 Tanh-26                   [-1, 76]               0
              STN_TPN-27          [-1, 3, 256, 256]               0
    ================================================================
    Total params: 118,196,114
    Trainable params: 118,196,114
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 147456.00
    Forward/backward pass size (MB): 83.58
    Params size (MB): 450.88
    Estimated Total Size (MB): 147990.47
    ----------------------------------------------------------------

### 2. Generator

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
       ReflectionPad2d-1          [-1, 3, 262, 262]               0
                Conv2d-2         [-1, 64, 256, 256]           9,472
        InstanceNorm2d-3         [-1, 64, 256, 256]               0
                  ReLU-4         [-1, 64, 256, 256]               0
                Conv2d-5        [-1, 128, 128, 128]          73,856
        InstanceNorm2d-6        [-1, 128, 128, 128]               0
                  ReLU-7        [-1, 128, 128, 128]               0
                Conv2d-8          [-1, 256, 64, 64]         295,168
        InstanceNorm2d-9          [-1, 256, 64, 64]               0
                 ReLU-10          [-1, 256, 64, 64]               0
      ReflectionPad2d-11          [-1, 256, 66, 66]               0
               Conv2d-12          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-13          [-1, 256, 64, 64]               0
                 ReLU-14          [-1, 256, 64, 64]               0
      ReflectionPad2d-15          [-1, 256, 66, 66]               0
               Conv2d-16          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-17          [-1, 256, 64, 64]               0
        ResidualBlock-18          [-1, 256, 64, 64]               0
      ReflectionPad2d-19          [-1, 256, 66, 66]               0
               Conv2d-20          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-21          [-1, 256, 64, 64]               0
                 ReLU-22          [-1, 256, 64, 64]               0
      ReflectionPad2d-23          [-1, 256, 66, 66]               0
               Conv2d-24          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-25          [-1, 256, 64, 64]               0
        ResidualBlock-26          [-1, 256, 64, 64]               0
      ReflectionPad2d-27          [-1, 256, 66, 66]               0
               Conv2d-28          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-29          [-1, 256, 64, 64]               0
                 ReLU-30          [-1, 256, 64, 64]               0
      ReflectionPad2d-31          [-1, 256, 66, 66]               0
               Conv2d-32          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-33          [-1, 256, 64, 64]               0
        ResidualBlock-34          [-1, 256, 64, 64]               0
      ReflectionPad2d-35          [-1, 256, 66, 66]               0
               Conv2d-36          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-37          [-1, 256, 64, 64]               0
                 ReLU-38          [-1, 256, 64, 64]               0
      ReflectionPad2d-39          [-1, 256, 66, 66]               0
               Conv2d-40          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-41          [-1, 256, 64, 64]               0
        ResidualBlock-42          [-1, 256, 64, 64]               0
      ReflectionPad2d-43          [-1, 256, 66, 66]               0
               Conv2d-44          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-45          [-1, 256, 64, 64]               0
                 ReLU-46          [-1, 256, 64, 64]               0
      ReflectionPad2d-47          [-1, 256, 66, 66]               0
               Conv2d-48          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-49          [-1, 256, 64, 64]               0
        ResidualBlock-50          [-1, 256, 64, 64]               0
      ReflectionPad2d-51          [-1, 256, 66, 66]               0
               Conv2d-52          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-53          [-1, 256, 64, 64]               0
                 ReLU-54          [-1, 256, 64, 64]               0
      ReflectionPad2d-55          [-1, 256, 66, 66]               0
               Conv2d-56          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-57          [-1, 256, 64, 64]               0
        ResidualBlock-58          [-1, 256, 64, 64]               0
      ReflectionPad2d-59          [-1, 256, 66, 66]               0
               Conv2d-60          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-61          [-1, 256, 64, 64]               0
                 ReLU-62          [-1, 256, 64, 64]               0
      ReflectionPad2d-63          [-1, 256, 66, 66]               0
               Conv2d-64          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-65          [-1, 256, 64, 64]               0
        ResidualBlock-66          [-1, 256, 64, 64]               0
      ReflectionPad2d-67          [-1, 256, 66, 66]               0
               Conv2d-68          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-69          [-1, 256, 64, 64]               0
                 ReLU-70          [-1, 256, 64, 64]               0
      ReflectionPad2d-71          [-1, 256, 66, 66]               0
               Conv2d-72          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-73          [-1, 256, 64, 64]               0
        ResidualBlock-74          [-1, 256, 64, 64]               0
      ReflectionPad2d-75          [-1, 256, 66, 66]               0
               Conv2d-76          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-77          [-1, 256, 64, 64]               0
                 ReLU-78          [-1, 256, 64, 64]               0
      ReflectionPad2d-79          [-1, 256, 66, 66]               0
               Conv2d-80          [-1, 256, 64, 64]         590,080
       InstanceNorm2d-81          [-1, 256, 64, 64]               0
        ResidualBlock-82          [-1, 256, 64, 64]               0
      ConvTranspose2d-83        [-1, 128, 128, 128]         295,040
       InstanceNorm2d-84        [-1, 128, 128, 128]               0
                 ReLU-85        [-1, 128, 128, 128]               0
      ConvTranspose2d-86         [-1, 64, 256, 256]          73,792
       InstanceNorm2d-87         [-1, 64, 256, 256]               0
                 ReLU-88         [-1, 64, 256, 256]               0
      ReflectionPad2d-89         [-1, 64, 262, 262]               0
               Conv2d-90          [-1, 3, 256, 256]           9,411
                 Tanh-91          [-1, 3, 256, 256]               0
         GuidedFilter-92          [-1, 3, 256, 256]               0
    ================================================================
    Total params: 11,378,179
    Trainable params: 11,378,179
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.75
    Forward/backward pass size (MB): 936.73
    Params size (MB): 43.40
    Estimated Total Size (MB): 980.88
    ----------------------------------------------------------------

### 3. Discriminator

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 128, 128]           3,136
             LeakyReLU-2         [-1, 64, 128, 128]               0
               Dropout-3         [-1, 64, 128, 128]               0
                Conv2d-4          [-1, 128, 64, 64]         131,200
        InstanceNorm2d-5          [-1, 128, 64, 64]               0
             LeakyReLU-6          [-1, 128, 64, 64]               0
               Dropout-7          [-1, 128, 64, 64]               0
                Conv2d-8          [-1, 256, 32, 32]         524,544
        InstanceNorm2d-9          [-1, 256, 32, 32]               0
            LeakyReLU-10          [-1, 256, 32, 32]               0
               Conv2d-11            [-1, 1, 31, 31]           4,097
    ================================================================
    Total params: 662,977
    Trainable params: 662,977
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.75
    Forward/backward pass size (MB): 46.01
    Params size (MB): 2.53
    Estimated Total Size (MB): 49.29
    ----------------------------------------------------------------
