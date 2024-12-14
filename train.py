import torch
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
from collections import OrderedDict

from functools import reduce
from thop import profile
import copy
import os 

from trainer import Trainer




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="config/kwai_simpleVQA.yml", help="Initial option file (coarse training)"
    )

    parser.add_argument(
        "-t", "--train_set", type=str, default="train", help="target_set"
    )
    parser.add_argument(
        "-t1", "--test_set", type=str, default="val-ltest", help="target_set"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default="./checkpoint/", help="target_set"
    )

    parser.add_argument('--gpu_id',type=str,default='1,2')


    args = parser.parse_args()

    # 加载初始配置（粗略训练配置）
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print("Loaded coarse training configuration:", opt)

    # 创建 checkpoint 目录
    if not os.path.exists(args.resume):
                os.makedirs(args.resume)

    # 初始化 Trainer 对象
    trainer=Trainer(args, opt)
    trainer.build_optimizer()

    # 设置粗细训练的切换点
    coarse_epochs = opt.get("num_epochs", 5)  # 使用配置中的粗略阶段 epoch 数
    fine_config_path = "config/kwai_fine_simpleVQA.yml"  # 精细阶段配置文件路径

    # 开始训练循环
    for epoch in range(coarse_epochs):
        print(f"End-to-end Epoch {epoch + 1}/{coarse_epochs}:")
        bests, bests_n = trainer.train_eval_all_epoches(epoch)

        # 打印模型的验证精度
        if coarse_epochs >= 0:
            print(
                f"""
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[0]:.4f}
                    PLCC:  {bests[1]:.4f}
                    KROCC: {bests[2]:.4f}
                    RMSE:  {bests[3]:.4f}."""
            )
            print(
                f"""
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[0]:.4f}
                    PLCC:  {bests_n[1]:.4f}
                    KROCC: {bests_n[2]:.4f}
                    RMSE:  {bests_n[3]:.4f}."""
            )

    # 加载精细阶段配置并更新训练轮数
    with open(fine_config_path, "r") as f:
        fine_opt = yaml.safe_load(f)
    print("Switching to fine training configuration:", fine_opt)
    trainer.update_config(fine_opt)
    trainer.build_optimizer()  # 更新优化器
    fine_epochs = fine_opt.get("num_epochs", 10)  # 从精细阶段配置获取新的 epoch 数

    # 精细阶段训练循环
    for epoch in range(fine_epochs):
        print(f"Fine Training Epoch {epoch + 1}/{fine_epochs}:")
        bests, bests_n = trainer.train_eval_all_epoches(epoch)

        # 打印精细阶段模型的验证精度
        if fine_epochs >= 0:
            print(
                f"""
                    the best validation accuracy of the model-s in fine stage is as follows:
                    SROCC: {bests[0]:.4f}
                    PLCC:  {bests[1]:.4f}
                    KROCC: {bests[2]:.4f}
                    RMSE:  {bests[3]:.4f}."""
            )
            print(
                f"""
                    the best validation accuracy of the model-n in fine stage is as follows:
                    SROCC: {bests_n[0]:.4f}
                    PLCC:  {bests_n[1]:.4f}
                    KROCC: {bests_n[2]:.4f}
                    RMSE:  {bests_n[3]:.4f}."""
            )

if __name__ == "__main__":
    main()
