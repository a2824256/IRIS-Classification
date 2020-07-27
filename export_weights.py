from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import csv
from paddle.utils.plot import Ploter
import sys


params_dirname = "./my_paddle_model"
place = fluid.CUDAPlace(0)
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()


with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)

    fc1_w = np.array(fluid.global_scope().find_var('fc1.w_0').get_tensor())
    fc1_b = np.array(fluid.global_scope().find_var('fc1.b_0').get_tensor())
    fc2_w = np.array(fluid.global_scope().find_var('fc2.w_0').get_tensor())
    fc2_b = np.array(fluid.global_scope().find_var('fc2.b_0').get_tensor())
    fc3_w = np.array(fluid.global_scope().find_var('fc3.w_0').get_tensor())
    fc3_b = np.array(fluid.global_scope().find_var('fc3.b_0').get_tensor())
    soft_w = np.array(fluid.global_scope().find_var('soft.w_0').get_tensor())
    soft_b = np.array(fluid.global_scope().find_var('soft.b_0').get_tensor())