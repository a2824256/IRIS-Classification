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
prediction = None
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
    LABEL = fluid.data(name='label', shape=[None, 1], dtype='int64')
    data_loader = fluid.io.DataLoader.from_generator(feed_list=[INPUT, LABEL], capacity=3, iterable=True)
    hidden = fluid.layers.fc(name='fc1', input=INPUT, size=10, act='relu')
    hidden = fluid.layers.fc(name='fc2', input=hidden, size=20, act='relu')
    hidden = fluid.layers.fc(name='fc3', input=hidden, size=10, act='relu')
    prediction = fluid.layers.fc(name='soft', input=hidden, size=3, act='softmax')
    loss = fluid.layers.mean(fluid.layers.cross_entropy(input=prediction, label=LABEL))
    acc = fluid.layers.accuracy(input=prediction, label=LABEL)
    return loss, data_loader, acc


data_pretreatment()
train_prog = fluid.Program()
train_startup = fluid.Program()
INPUT = fluid.data(name='input', shape=[None, 4], dtype='float32')
LABEL = fluid.data(name='label', shape=[None, 1], dtype='int64')
with fluid.program_guard(train_prog, train_startup):
    with fluid.unique_name.guard():
        train_loss, train_loader, acc = network()
        adam = fluid.optimizer.Adam(learning_rate=0.01)
        adam.minimize(train_loss)

test_prog = fluid.Program()
test_startup = fluid.Program()

# 定义预测网络
with fluid.program_guard(test_prog, test_startup):
    # Use fluid.unique_name.guard() to share parameters with train network
    with fluid.unique_name.guard():
        test_loss, test_loader, acc = network()

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(train_startup)
exe.run(test_startup)

# Compile programs
train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=train_loss.name)
test_prog = fluid.CompiledProgram(test_prog).with_data_parallel(share_vars_from=train_prog)
train_loader.set_sample_generator(sample_reader, batch_size=32, places=PLACE)
test_loader.set_sample_generator(sample_reader, batch_size=32, places=PLACE)

def run_iterable(program, exe, loss, data_loader, acc):
    for data in data_loader():
        loss_value = exe.run(program=program, feed=data, fetch_list=[loss])
        acc_value = exe.run(program=program, feed=data, fetch_list=[acc])
        print('loss is {}, acc is {}'.format(loss_value, acc_value))

for epoch_id in range(10):
    print("----train start----")
    run_iterable(train_prog, exe, train_loss, train_loader, acc)
    param_path = "./my_paddle_model"
    # fluid.io.save_params(executor=exe, dirname=param_path, main_program=train_startup)
    # print(type(prediction))
    fluid.io.save_inference_model(SAVE_PATH, ['input'], [prediction], exe)
    print("----test start----")
    run_iterable(test_prog, exe, test_loss, test_loader, acc)


