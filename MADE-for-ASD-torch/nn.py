#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders training and fine-tuning.

Usage:
  nn_pytorch.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn_pytorch.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_classif

from docopt import docopt
from utils import (load_phenotypes, format_config, hdf5_handler, load_fold,
                          sparsity_penalty, reset, to_softmax)

from model import ae, nn


def run_autoencoder1(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, code_size=1000):
    """
    Run the first autoencoder.
    It takes the original data dimensionality and compresses it into `code_size`
    """

    # 检查模型是否已存在
    if os.path.isfile(model_path):
        return

    # 超参数
    learning_rate = 0.0001
    sparse = True  # 添加稀疏性惩罚
    sparse_p = 0.2
    sparse_coeff = 0.5
    corruption = 0.7  # 数据损坏比例用于去噪
    ae_enc = torch.tanh  # 双曲正切
    ae_dec = None  # 线性激活

    training_iters = 700
    batch_size = 100
    n_classes = 2

    device = torch.device("cuda")
    
    # 将NumPy数组转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # 创建模型并添加稀疏性惩罚
    model = ae(X_train.shape[1], code_size, corruption=corruption, enc=ae_enc, dec=ae_dec).to(device)#X_train.shape[1]: 1000 code_size: 1000
    
    # 使用梯度下降优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 初始化一个非常大的代价用于模型选择
    prev_costs = np.array([9999999999] * 3)

    for epoch in range(training_iters):
        # 将训练集分成批次
        n_batches = (len(X_train) + batch_size - 1) // batch_size  # 向上取整
        batches = range(n_batches)
        costs = np.zeros((len(batches), 3))

        for ib in batches:
            # 计算当前批次的起始和结束索引
            from_i = ib * batch_size
            to_i = min((ib + 1) * batch_size, len(X_train))

            # 选择当前批次
            batch_xs = X_train_tensor[from_i:to_i]

            # 训练和获取训练代价
            optimizer.zero_grad()
            encode, decode = model(batch_xs)
            cost_train = model.get_cost(batch_xs, decode)
            
            # 如果需要，添加稀疏性惩罚
            if sparse:
                cost_train = cost_train + sparsity_penalty(encode, sparse_p, sparse_coeff)
                
            cost_train.backward()
            optimizer.step()

            # 计算验证代价
            with torch.no_grad():
                encode_valid, decode_valid = model(X_valid_tensor)
                cost_valid = model.get_cost(X_valid_tensor, decode_valid)
                if sparse:
                    cost_valid = cost_valid + sparsity_penalty(encode_valid, sparse_p, sparse_coeff)

                # 计算测试代价
                encode_test, decode_test = model(X_test_tensor)
                cost_test = model.get_cost(X_test_tensor, decode_test)
                if sparse:
                    cost_test = cost_test + sparsity_penalty(encode_test, sparse_p, sparse_coeff)

            costs[ib] = [cost_train.item(), cost_valid.item(), cost_test.item()]

        # 计算所有批次的平均代价
        costs = costs.mean(axis=0)
        cost_train, cost_valid, cost_test = costs

        # 打印训练信息
        # print(
        #     f"Exp={experiment}, Model=ae1, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}"
        # )

        # 如果验证代价降低，保存更好的模型
        if cost_valid < prev_costs[1]:
            # print("Saving better model")
            # 确保模型目录存在
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)
            prev_costs = costs
        # else:
        #     print()


def run_autoencoder2(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, prev_model_path,
                     code_size=600, prev_code_size=1000):
    """
    Run the second autoencoder.
    It takes the dimensionality from first autoencoder and compresses it into the new `code_size`
    Firstly, we need to convert original data to the new projection from autoencoder 1.
    """

    if os.path.isfile(model_path):
        return

    device = torch.device("cuda")
    
    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # 使用第一个自编码器转换原始数据
    prev_model = ae(X_train.shape[1], prev_code_size, #1000   1000
                   corruption=0.0,  # 禁用数据损坏
                   enc=torch.tanh, dec=None).to(device)
    
    if os.path.isfile(prev_model_path):
        prev_model.load_state_dict(torch.load(prev_model_path))
    
    with torch.no_grad():
        # 转换训练、验证和测试集
        X_train_enc, _ = prev_model(X_train_tensor)
        X_valid_enc, _ = prev_model(X_valid_tensor)
        X_test_enc, _ = prev_model(X_test_tensor)
        
        # 转回NumPy数组
        X_train = X_train_enc.cpu().numpy()
        X_valid = X_valid_enc.cpu().numpy()
        X_test = X_test_enc.cpu().numpy()
    
    del prev_model
    
    # 重新设置随机种子
    reset()
    
    # 超参数
    learning_rate = 0.0001
    corruption = 0.9
    ae_enc = torch.tanh
    ae_dec = None

    training_iters = 1000
    batch_size = 10
    n_classes = 2
    
    # 将转换后的数据再次转为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # 加载模型
    model = ae(prev_code_size, code_size, corruption=corruption, enc=ae_enc, dec=ae_dec).to(device)#1000    600
    
    # 使用梯度下降优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 初始化一个非常大的代价用于模型选择
    prev_costs = np.array([9999999999] * 3)

    # 迭代训练
    for epoch in range(training_iters):
        # 将训练集分成批次
        n_batches = (len(X_train) + batch_size - 1) // batch_size  # 向上取整
        batches = range(n_batches)
        costs = np.zeros((len(batches), 3))

        for ib in batches:
            # 计算当前批次的起始和结束索引
            from_i = ib * batch_size
            to_i = min((ib + 1) * batch_size, len(X_train))

            # 选择当前批次
            batch_xs = X_train_tensor[from_i:to_i]

            # 训练和获取训练代价
            optimizer.zero_grad()
            encode, decode = model(batch_xs)
            cost_train = model.get_cost(batch_xs, decode)
            cost_train.backward()
            optimizer.step()

            # 计算验证代价
            with torch.no_grad():
                encode_valid, decode_valid = model(X_valid_tensor)
                cost_valid = model.get_cost(X_valid_tensor, decode_valid)

                # 计算测试代价
                encode_test, decode_test = model(X_test_tensor)
                cost_test = model.get_cost(X_test_tensor, decode_test)

            costs[ib] = [cost_train.item(), cost_valid.item(), cost_test.item()]

        # 计算所有批次的平均代价
        costs = costs.mean(axis=0)
        cost_train, cost_valid, cost_test = costs

        # 打印训练信息
        # print(
        #     f"Exp={experiment}, Model=ae2, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}"
        # )

        # 如果验证代价降低，保存更好的模型
        if cost_valid < prev_costs[1]:
            # print("Saving better model")
            # 确保模型目录存在
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)
            prev_costs = costs
        # else:
        #     print()


def run_finetuning(experiment,
                   X_train, y_train, X_valid, y_valid, X_test, y_test,
                   model_path, prev_model_1_path, prev_model_2_path,
                   code_size_1=1000, code_size_2=600):
    """
    Run the pre-trained NN for fine-tuning, using first and second autoencoders' weights
    """

    # 超参数
    learning_rate = 0.0005
    dropout_1 = 0.5
    dropout_2 = 0.8
    dropout_3 = 0.3
    initial_momentum = 0.1
    final_momentum = 0.9  # 沿着训练增加动量以避免波动
    saturate_momentum = 100

    training_iters = 200
    start_saving_at = 20
    batch_size = 10
    n_classes = 2

    if os.path.isfile(model_path):
        return

    device = torch.device("cuda")
    
    # 将输出转换为one-hot编码
    y_train_oh = np.array([to_softmax(n_classes, y) for y in y_train])
    y_valid_oh = np.array([to_softmax(n_classes, y) for y in y_valid])
    y_test_oh = np.array([to_softmax(n_classes, y) for y in y_test])
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train_oh).to(device)
    y_valid_tensor = torch.FloatTensor(y_valid_oh).to(device)
    y_test_tensor = torch.FloatTensor(y_test_oh).to(device)


# 加载预训练的编码器权重
    # 替换自定义的load_ae_encoder函数
    # 加载第一个自编码器


    ae1_model = ae(X_train.shape[1] - 2, code_size_1, corruption=0.0, enc=torch.tanh, dec=None).to(device)#1000   1000
    
    ae1_model.load_state_dict(torch.load(prev_model_1_path))
    
    # 加载第二个自编码器
    ae2_model = ae(code_size_1, code_size_2, corruption=0.0, enc=torch.tanh, dec=None).to(device)
    ae2_model.load_state_dict(torch.load(prev_model_2_path))
    
    # 使用编码器权重初始化NN模型
    layers = [
        {"size": code_size_1, "actv": torch.tanh},
        {"size": code_size_2, "actv": torch.tanh},
        {"size": 100, "actv": torch.tanh},
    ]
    
    # 从模型中提取权重和偏置
    # 获取第一个编码器的权重和偏置
    ae1_weights = ae1_model.W_enc.data.cpu().numpy()
    ae1_bias = ae1_model.b_enc.data.cpu().numpy()
    
    # 获取第二个编码器的权重和偏置
    ae2_weights = ae2_model.W_enc.data.cpu().numpy()
    ae2_bias = ae2_model.b_enc.data.cpu().numpy()
    
    # 确保使用与TF相同的权重初始化方法
    init = [
        {"W": np.vstack([ae1_weights, np.zeros((2, ae1_weights.shape[1]))]), "b": ae1_bias},
        {"W": ae2_weights, "b": ae2_bias},
        {"W": (np.random.randn(600, 100)/10000).astype(np.float32), "b": ae2_bias[:100]},
    ]
    
    model = nn(X_train.shape[1], n_classes, layers, init).to(device)
    # 使用带动量的梯度下降优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=initial_momentum)
    
    # 初始化一个非常大的代价和低准确率用于模型选择
    prev_costs = np.array([9999999999] * 3)
    prev_accs = np.array([0.0] * 3)

    # 迭代训练
    for epoch in range(training_iters):
        # 将训练集分成批次
        # 确保与TF版本完全相同的批次划分
        n_batches = (len(X_train) + batch_size - 1) // batch_size  # 向上取整
        batches = range(n_batches)
        costs = np.zeros((len(batches), 3))
        accs = np.zeros((len(batches), 3))

        # 计算动量饱和度
        alpha = float(epoch) / float(saturate_momentum)
        if alpha < 0.:
            alpha = 0.
        if alpha > 1.:
            alpha = 1.
        momentum = initial_momentum * (1 - alpha) + alpha * final_momentum
        
        # 更新优化器的动量参数
        for param_group in optimizer.param_groups:
            param_group['momentum'] = momentum

        for ib in batches:
            # 计算当前批次的起始和结束索引
            from_i = ib * batch_size
            to_i = min((ib + 1) * batch_size, len(X_train))  # 确保不超出数据集边界

            # 选择当前批次
            batch_xs, batch_ys = X_train_tensor[from_i:to_i, :], y_train_tensor[from_i:to_i, :]

            # 训练和获取训练代价和准确率
            optimizer.zero_grad()
            y_hat, output, activations = model(batch_xs, dropout_rates=[dropout_1, dropout_2, dropout_3])
            cost_train = model.get_cost(y_hat, batch_ys)
            cost_train.backward()
            optimizer.step()
            
            # 计算训练准确率
            pred = torch.argmax(output, dim=1)
            target = torch.argmax(batch_ys, dim=1)
            acc_train = torch.mean((pred == target).float()).item()

            # 计算验证代价和准确率
            with torch.no_grad():
                y_hat_valid, output_valid, _ = model(X_valid_tensor, dropout_rates=[1.0, 1.0, 1.0])
                cost_valid = model.get_cost(y_hat_valid, y_valid_tensor)
                pred_valid = torch.argmax(output_valid, dim=1)
                target_valid = torch.argmax(y_valid_tensor, dim=1)
                acc_valid = torch.mean((pred_valid == target_valid).float()).item()

                # 计算测试代价和准确率
                y_hat_test, output_test, _ = model(X_test_tensor, dropout_rates=[1.0, 1.0, 1.0])
                cost_test = model.get_cost(y_hat_test, y_test_tensor)
                pred_test = torch.argmax(output_test, dim=1)
                target_test = torch.argmax(y_test_tensor, dim=1)
                acc_test = torch.mean((pred_test == target_test).float()).item()

            costs[ib] = [cost_train.item(), cost_valid.item(), cost_test.item()]
            accs[ib] = [acc_train, acc_valid, acc_test]

        # 计算所有批次的平均代价和准确率
        costs = costs.mean(axis=0)
        cost_train, cost_valid, cost_test = costs
        
        accs = accs.mean(axis=0)
        acc_train, acc_valid, acc_test = accs

        # 打印训练信息
        # print(
        #     f"Exp={experiment}, Model=mlp, Iter={epoch:5d}, Acc={acc_train:.6f} {acc_valid:.6f} {acc_test:.6f}, Momentum={momentum:.6f}"
        # )

        # 如果验证准确率提高且已过初始波动期，保存更好的模型
        if acc_valid > prev_accs[1] and epoch > start_saving_at:
            #print("Saving better model")
            # 确保模型目录存在
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)
            prev_accs = accs
            prev_costs = costs
        # else:
        #     print()

    # 返回最终结果
    with torch.no_grad():
        _, output_test, _ = model(X_test_tensor, dropout_rates=[1.0, 1.0, 1.0])
        return y_test_tensor.cpu().numpy(), output_test.cpu().numpy()


def run_nn(hdf5, experiment, code_size_1, code_size_2):
    exp_storage = hdf5["experiments"][experiment]
    y_test_s = []
    y_pred_s = []
    for fold in exp_storage:
        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })
        
        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)

        X_all = np.vstack((X_train, X_valid, X_test))
        # print(X_all.shape)
        
        age_sex = X_all[:, -2:]
        X_all = X_all[:, :-2]
        y_all = np.concatenate((np.array(y_train), np.array(y_valid), np.array(y_test)), axis=0)
        # print(y_all.shape)

        ks = 0
        if X_all.shape[1] < 10000:
            ks = 1000
        else:
            ks = 3000
        X_new = SelectKBest(f_classif, k=ks).fit_transform(X_all, y_all)
        # print(X_new.shape)

        train = X_train.shape[0]
        valid = X_valid.shape[0]
        test = X_test.shape[0]

        X_train = X_new[:train]
        X_valid = X_new[train:train+valid]
        X_test = X_new[train+valid:train+valid+test]

        X_pheno = np.concatenate((X_new, age_sex), axis=1)
        # print(X_pheno.shape)

        X_train_2 = X_pheno[:train]
        X_valid_2 = X_pheno[train:train+valid]
        X_test_2 = X_pheno[train+valid:train+valid+test]

        # print(X_test_2.shape)
        
        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.pt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.pt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.pt", {
            "experiment": experiment_cv,
        })

        reset()

        # 运行第一个自编码器
        run_autoencoder1(experiment_cv,
                         X_train, y_train, X_valid, y_valid, X_test, y_test,
                         model_path=ae1_model_path,
                         code_size=code_size_1)

        reset()

        # 运行第二个自编码器
        run_autoencoder2(experiment_cv,
                         X_train, y_train, X_valid, y_valid, X_test, y_test,
                         model_path=ae2_model_path,
                         prev_model_path=ae1_model_path,
                         prev_code_size=code_size_1,
                         code_size=code_size_2)

        reset()

        # 运行使用预训练自编码器的多层NN
        y_test, y_pred = run_finetuning(experiment_cv,
                                        X_train_2, y_train, X_valid_2, y_valid, X_test_2, y_test,
                                        model_path=nn_model_path,
                                        prev_model_1_path=ae1_model_path,
                                        prev_model_2_path=ae2_model_path,
                                        code_size_1=code_size_1,
                                        code_size_2=code_size_2)
        reset()

        y_test_s.append(y_test), y_pred_s.append(y_pred)

    return y_test_s, y_pred_s


if __name__ == "__main__":
    reset()

    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler(bytes("./data/abide.hdf5", encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []

    for derivative in derivatives:
        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)]

        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                if site == 'NYU':
                    site_config = {"site": site}
                    experiments += [
                        format_config("{derivative}_leavesiteout-{site}",
                                    config, site_config)
                    ]

    # 第一个自编码器瓶颈
    code_size_1 = 1000

    # 第二个自编码器瓶颈
    code_size_2 = 600

    experiments = sorted(experiments)

    all_y_test = []
    all_y_pred = []
    turn = 0
    
    for experiment in experiments:
        print(experiment)
        if turn == 0:
            # 第一个实验
            y_test_s, y_pred_s = run_nn(hdf5, experiment, code_size_1, code_size_2)
            all_y_test = y_test_s
            all_y_pred = y_pred_s
        else:
            # 后续实验
            _, pred_s = run_nn(hdf5, experiment, code_size_1, code_size_2)
            
            # 对每个交叉验证折应用权重
            for i in range(len(pred_s)):
                if turn == 1:
                    all_y_pred[i] += 0.5 * pred_s[i]
                elif turn == 2:
                    all_y_pred[i] += 0.7 * pred_s[i]
        
        turn += 1

    # 计算每个交叉验证折的准确率
    fold_accuracies = []
    for i in range(len(all_y_test)):
        y_test = all_y_test[i]
        y_pred = all_y_pred[i]
        correct = np.equal(np.argmax(y_pred, 1), np.argmax(y_test, 1))
        accuracy = np.mean(correct)
        fold_accuracies.append(accuracy)
        print(f"Fold {i} accuracy: {accuracy:.6f}")
    
    # 计算平均准确率
    mean_accuracy = np.mean(fold_accuracies)
    print(f"平均准确率: {mean_accuracy:.6f}")