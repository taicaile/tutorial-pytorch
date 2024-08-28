from os import write
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

# ---- summarywriter test ----
flag = False
# flag = True
if flag:
    writer = SummaryWriter(comment="test tensorboard")

    for x in range(100):
        writer.add_scalar("y=2x", x*2, x)
        writer.add_scalar("y=pow(2,x)", 2**x, x)
        writer.add_scalars('data/scalar_group', {'xsinx': x*np.sin(x),
                                                'xcosx':x*np.cos(x),
                                                'arctanx': np.arctan(x)}, x)
    writer.close()

# ---- summarywriter parameters
flag = False
# flag = True
if flag:
    # comment, 不指定log_dir时，文件夹的后缀名
    # writer = SummaryWriter(log_dir="writer_logs", comment="_comment", filename_suffix='_suffix')
    writer = SummaryWriter(comment="_comment", filename_suffix='_suffix')

    for x in range(100):
        writer.add_scalar("y=2x", x*2, x)
        writer.add_scalar("y=pow(2,x)", 2**x, x)
        writer.add_scalars('data/scalar_group', {'xsinx': x*np.sin(x),
                                                'xcosx':x*np.cos(x),
                                                'arctanx': np.arctan(x)}, x)
    writer.close()

# ---- histogram ----
flag = False
flag = True
if flag:
    # comment, 不指定log_dir时，文件夹的后缀名
    # writer = SummaryWriter(log_dir="writer_logs", comment="_comment", filename_suffix='_suffix')
    writer = SummaryWriter(comment="_comment", filename_suffix='_suffix')

    for x in range(2):

        np.random.seed(x)

        data_union = np.arange(100)
        data_normal = np.random.normal(size=1000)

        writer.add_histogram("union",data_union,x)
        writer.add_histogram("normal", data_normal,x)

        plt.subplot(121).hist(data_union, label='union')
        plt.subplot(122).hist(data_normal, label='normal')

        plt.legend()
        plt.show()

    writer.close()
