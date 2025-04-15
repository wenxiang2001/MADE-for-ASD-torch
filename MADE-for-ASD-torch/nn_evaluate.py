#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from docopt import docopt
from model import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_fold)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif


def nn_results(hdf5, experiment, code_size_1, code_size_2):

    exp_storage = hdf5["experiments"][experiment]

    n_classes = 2

    results = []

    count = 0
    for fold in exp_storage:
        if count == 1:
            break
        count += 1
        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })

        
        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)


        X_all = np.vstack((X_train, X_valid, X_test))
        y_all = np.concatenate((np.array(y_train), np.array(y_valid), np.array(y_test)), axis=0)

        # 特征选择
        ks = 0
        if X_all.shape[1] < 10000:
            ks = 1000
        else:
            ks = 3000    
        X_new = SelectKBest(f_classif, k=ks).fit_transform(X_all[:, :-2], y_all)

        # 合并特征选择后的特征和最后两列（年龄和性别）
        X_new = np.concatenate((X_new, X_all[:, -2:]), axis=1)

        train = X_train.shape[0]
        valid = X_valid.shape[0]
        test = X_test.shape[0]

        X_train = X_new[:train]
        X_valid = X_new[train:train+valid]
        X_test = X_new[train+valid:train+valid+test]

        # 转换为one-hot编码
        y_test = np.array([to_softmax(n_classes, y) for y in y_all])

        # 模型路径
        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.pt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.pt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.pt", {
            "experiment": experiment_cv,
        })

        try:
            # 设备选择
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 转换为PyTorch张量
            X_new_tensor = torch.FloatTensor(X_new).to(device)
            
            # 定义模型结构
            layers = [
                {"size": 1000, "actv": torch.tanh},
                {"size": 600, "actv": torch.tanh},
                {"size": 100, "actv": torch.tanh},
            ]
            
            # 创建模型
            model = nn(X_test.shape[1], n_classes, layers)
            model.to(device)
            
            # 加载预训练模型
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(nn_model_path))
            else:
                model.load_state_dict(torch.load(nn_model_path, map_location=torch.device('cpu')))
            
            # 设置为评估模式
            model.eval()
            
            # 前向传播
            with torch.no_grad():
                _, output, _ = model(X_new_tensor, dropout_rates=[1.0, 1.0, 1.0])
                
                # 转换为NumPy数组
                output = output.cpu().numpy()
                
                # 预测类别和真实类别
                y_pred = np.argmax(output, axis=1)
                y_true = np.argmax(y_test, axis=1)
                
                # 保存匹配的样本
                X_sub = []
                y_sub = []
                
                for i in range(X_new.shape[0]):
                    if y_pred[i] == y_true[i]:
                        X_sub.append(X_all[i])
                        y_sub.append(y_pred[i])
                
                X_sub = np.array(X_sub)
                y_sub = np.array(y_sub)
                
                # 保存匹配的样本
                np.save('X_sub_without_NYU.npy', X_sub)
                np.save('y_sub_without_NYU.npy', y_sub)
                
                # 计算指标
                [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
                specificity = TN/(FP+TN)
                precision = TP/(TP+FP)
                sensivity = TP/(TP+FN)
                
                accuracy = accuracy_score(y_true, y_pred)
                fscore = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)
                
                results.append([accuracy, precision, fscore, sensivity, specificity, roc_auc])
        finally:
            reset()

    return [experiment] + np.mean(results, axis=0).tolist()

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pd.set_option("display.expand_frame_repr", False)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler(bytes("./data/abide.hdf5",encoding="utf8"), "a")

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
                if site=='NYU':
                  site_config = {"site": site}
                  experiments += [
                      format_config("{derivative}_leavesiteout-{site}",
                                    config, site_config)
                  ]

        if arguments["--NYU-site-out"]:
            experiments += [format_config("{derivative}_leavesiteout-NYU", config)]


    # First autoencoder bottleneck
    code_size_1 = 1000 # not used

    # Second autoencoder bottleneck
    code_size_2 = 600 # not used

    results = []

    experiments = sorted(experiments)
    print(experiments)
    for experiment in experiments:
        print(experiment)
        results.append(nn_results(hdf5, experiment, code_size_1, code_size_2))

    cols = ["Exp", "Accuracy", "Precision", "F1-score", "Sensivity", "Specificity", "ROC-AUC"]
    df = pd.DataFrame(results, columns=cols)

    print('aaa',df[cols] \
        .sort_values(["Exp"]) \
        .reset_index())