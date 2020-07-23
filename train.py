from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import csv
from paddle.utils.plot import Ploter

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


def train(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1  # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated]


# data pretreatment
data_pretreatment()

# network
INPUT = fluid.data(name='input', shape=[None, 4], dtype='float32')
LABEL = fluid.data(name='label', shape=[None, 1], dtype='int64')
data_loader = fluid.io.DataLoader.from_generator(feed_list=[INPUT, LABEL], capacity=3, iterable=True)
hidden = fluid.layers.fc(name='fc1', input=INPUT, size=10, act='relu')
hidden = fluid.layers.fc(name='fc2', input=hidden, size=20, act='relu')
hidden = fluid.layers.fc(name='fc3', input=hidden, size=10, act='relu')
prediction = fluid.layers.fc(name='soft', input=hidden, size=3, act='softmax')

# main program
main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program()

# loss
loss = fluid.layers.mean(fluid.layers.cross_entropy(input=prediction, label=LABEL))
acc = fluid.layers.accuracy(input=prediction, label=LABEL)

# test program
test_program = main_program.clone(for_test=True)
adam = fluid.optimizer.Adam(learning_rate=0.01)
adam.minimize(loss)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

num_epochs = 100

params_dirname = "./my_paddle_model"
feeder = fluid.DataFeeder(place=place, feed_list=[INPUT, LABEL])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)