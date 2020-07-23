from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import csv

ITERABLE = True
PLACE = fluid.cuda_places(0)
DATA = []
RES = []
FILE = "./iris.data"
SAVE_PATH = "./params"


def data_pretreatment():
    with open(FILE) as f:
        render = csv.reader(f)
        for row in render:
            if len(row) != 0:
                DATA.append([row[0], row[1], row[2], row[3]])
                RES.append([row[4]])


def sample_reader():
    for i in range(100):
        input = np.array(DATA).astype('float32')
        label = np.array(RES).astype('int64')
        yield input, label


def network():
    INPUT = fluid.data(name='input', shape=[None, 4], dtype='float32')
    hidden = fluid.layers.fc(name='fc1', input=INPUT, size=10, act='relu')
    hidden = fluid.layers.fc(name='fc2', input=hidden, size=20, act='relu')
    hidden = fluid.layers.fc(name='fc3', input=hidden, size=10, act='relu')
    prediction = fluid.layers.fc(name='soft', input=hidden, size=3, act='softmax')
    return prediction


data_pretreatment()
pred_prog = fluid.Program()
exe = fluid.Executor(PLACE)
exe.run(pred_prog)
pred_prog = fluid.CompiledProgram(pred_prog).with_data_parallel(share_vars_from=pred_prog)