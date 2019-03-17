#!/usr/bin/env python
#encoding:utf8

import paddle
import paddle.fluid as fluid
import logging
import time
import numpy as np
import os
import sys
import reader
import nets

use_cuda = True
learn_rate = 1e-4
model_save_path = "./output/models"
num_epochs = 100
batch_size = 256
train_sample_rate = 1
train_data_dir = "./data/train_data"
test_data_dir = "./data/test_data"

def main():
    logging.basicConfig(level = logging.NOTSET)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # train
    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(main_program = train_program, startup_program = train_startup):
        with fluid.unique_name.guard():
            [avg_cost, acc, predict, train_reader] = nets.bilstm_attention_classify_net(is_test = False, is_py_reader = True)
        optimizer = fluid.optimizer.Adam(learning_rate = learn_rate)
        optimizer.minimize(avg_cost)
    # test
    test_program = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(main_program = test_program, startup_program = test_startup):
        with fluid.unique_name.guard():
            [test_avg_cost, test_acc, test_predict, test_reader] = nets.bilstm_attention_classify_net(is_test = True, is_py_reader = True)

    infer_program = train_program.clone()
    fluid.memory_optimize(train_program)
    fluid.memory_optimize(test_program)
    
    executor = fluid.Executor(place)
    executor.run(train_startup)
    executor.run(test_startup)
    
    train_exe = fluid.ParallelExecutor(
            use_cuda = use_cuda, loss_name = avg_cost.name,
            main_program = train_program)
    device_count = train_exe.device_count
    logging.info("device count: %d" % device_count)
    logging.info("start train process ...")
    test_exe = fluid.ParallelExecutor(
            use_cuda = use_cuda, 
            share_vars_from = train_exe,
            main_program = test_program)
    train_reader.decorate_paddle_reader(
            paddle.batch(
                reader.reader(data_dir = train_data_dir, 
                    sample_rate = train_sample_rate),
                batch_size,
                drop_last = False))
    test_reader.decorate_paddle_reader(
            paddle.batch(
                reader.reader(data_dir = test_data_dir),
                batch_size,
                drop_last = False))
    
    for epoch_id in range(num_epochs):
        losses = []
        start_time = time.time()
        train_reader.start()
        iter = 0
        try:
            while True:
                avg_loss = train_exe.run([avg_cost.name])
                print("epoch: %d, iter: %d, loss: %f" % (epoch_id, iter, np.mean(avg_loss[0])))
                losses.append(np.mean(avg_loss[0]))
                iter += 1
        except fluid.core.EOFException:
            train_reader.reset()
            end_time = time.time()
            print("epoch: %d, loss: %f, used time: %d sec" 
                    % (epoch_id, np.mean(losses), end_time - start_time))
        logging.info("start test process ...")
        losses = []
        test_reader.start()
        try:
            while True:
                (avg_loss, avg_acc) = test_exe.run([test_avg_cost.name, test_acc.name])
                losses.append((np.mean(avg_loss[0]), np.mean(avg_acc[0])))
        except fluid.core.EOFException:
            test_reader.reset()
        print ("test at epoch: %d, loss: %f, acc: %f" 
                % (epoch_id, np.mean([i[0] for i in losses]), 
                    np.mean([i[1] for i in losses])))
    
        logging.info("start save process ...")
        model_path = os.path.join(model_save_path, str(epoch_id))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            # feed varload
            fluid.io.save_params(
                    executor = executor,
                    dirname = model_path,
                    main_program = infer_program)

if __name__ == '__main__':
    main()

