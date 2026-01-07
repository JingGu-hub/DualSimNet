import datetime
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from cleanlab.internal.constants import EPSILON

import torch.nn.functional as F

from utils.ExcelUtils import XlsReport


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')

    return filename

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

class CustomMultiStepLR:
    def __init__(self, optimizer, milestones, gammas):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gammas = gammas
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_epoch += 1
        for i, milestone in enumerate(self.milestones):
            if self.current_epoch == milestone:
                self.optimizer.param_groups[0]['lr'] *= self.gammas[i]

def normalize_rows(tensor):
    if tensor.dim() != 2:
        raise ValueError("输入张量必须是二维的")

    # 计算每一行的最小值和最大值
    min_values = tensor.min(dim=1, keepdim=True).values
    max_values = tensor.max(dim=1, keepdim=True).values

    # 避免除以零的情况
    range_values = max_values - min_values
    range_values[range_values == 0] = 1.0  # 如果范围为零，将其设置为1，避免除以零

    # 应用最大最小归一化公式
    normalized_tensor = (tensor - min_values) / range_values

    return normalized_tensor

def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(args, epoch, warm_up)

def write_matrix_to_excel(datas, model_names, file_name='results.xls'):
    # 文件的后缀名为xls
    file_path = os.path.join(os.getcwd(), file_name)
    xls = XlsReport(file_path)
    # 创建Excel对象
    xls.xlsOpenWorkbook()
    sheet = xls.xlsAddWorksheet('sheet')
    # Excel的标题
    xls.addWorksheetTitle(sheet, model_names)

    # 写入数据
    for i, data in enumerate(datas):
        xls.appendWorkshetData(sheet, data)
    xls.xlsCloseWorkbook(sheet)

def get_clean_loss_tensor_mask(loss_all, remember_rate):
    ind_1_sorted = torch.argsort(loss_all)
    mask_loss = torch.zeros(len(ind_1_sorted)).cuda()
    for i in range(int(len(ind_1_sorted) * remember_rate)):
        mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_loss

def gmm_divide(loss, loss_type=False):
    if loss_type == False:
        loss = -loss

    # fit a two-component GMM to the loss
    loss = loss.detach().cpu().numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)
    prob = gmm.predict_proba(loss)

    prob = prob[:, gmm.means_.argmin()]
    clean_mask = (prob > 0.5).astype(int)

    return torch.from_numpy(clean_mask).cuda()

def gmm_divide_loss(loss, loss_type=False):
    clean_mask = gmm_divide(loss, loss_type=loss_type)
    mean_loss = loss[clean_mask == 1].mean()

    return mean_loss, clean_mask

def f1_scores(output, y_true):
    target_pred = torch.argmax(output.data, axis=1)

    y_true = y_true.detach().cpu().numpy()
    y_pred = target_pred.detach().cpu().numpy()

    # 计算F1分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return f1_macro, f1_weighted, f1_micro

def static_class_num(target):
    num_classes = len(np.unique(target))

    class_num_list = []
    for i in range(num_classes):
        class_num = (target == i).sum()
        class_num_list.append(class_num)

    print(class_num_list)

def inter_intra_distance_compute(X, y):
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)

    # 类间散度
    Sb = 0
    for c in classes:
        Xc = X[y==c]
        mean_c = Xc.mean(axis=0)
        Sb += len(Xc) * np.sum((mean_c - overall_mean) ** 2)

    # 类内散度
    Sw = 0
    for c in classes:
        Xc = X[y==c]
        mean_c = Xc.mean(axis=0)
        Sw += np.sum((Xc - mean_c) ** 2)

    return Sb / Sw

def pairflip_penalty(args, logits):
    # regularization
    prior = torch.ones(args.num_classes) / args.num_classes
    prior = prior.cuda()
    pred_mean = torch.softmax(logits, dim=1).mean(0)
    penalty = torch.sum(prior * torch.log(prior / pred_mean))

    return penalty
