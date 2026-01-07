import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

import seaborn as sns

from utils.static_data import plot_heatmap


class double_cos:
    def __init__(self, args, train_size, class_rate=0.2):
        self.clas_use_index = []
        self.noise_scores_vetor = torch.zeros((train_size)).cuda()
        self.opt_pre_activations = torch.zeros((train_size, args.features_num)).cuda()
        self.class_rate = class_rate
        self.best_sim_scores = float('-inf')
        self.worse_off_sim_scores = float('inf')
        self.class_num_list = None

    def row_normalize(self, X):
        X_min, _ = X.min(dim=1, keepdim=True)  # 每行最小值
        X_max, _ = X.max(dim=1, keepdim=True)  # 每行最大值

        denom = X_max - X_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # 避免除以 0

        normalized = (X - X_min) / denom
        return normalized


    def cosine_similarities(self, pre_activations, class_i_index, class_j_index=None):
        vec = pre_activations[class_i_index, :]  # shape: (d,)
        vec_norm = torch.norm(vec, p=2)  # ||vec||

        if class_j_index is None:
            dot_products = torch.matmul(vec, pre_activations.T)  # shape: (N,)
            all_norms = torch.norm(pre_activations, p=2, dim=1)  # ||each row||
            cos_similarities = dot_products / (vec_norm * all_norms + 1e-10)
        else:
            other_vecs = pre_activations[class_j_index, :]  # shape: (k, d)
            dot_products = torch.matmul(vec, other_vecs.T)  # shape: (k,)
            all_norms = torch.norm(other_vecs, p=2, dim=1)  # ||each row||
            cos_similarities = dot_products / (vec_norm * all_norms + 1e-10)

        return cos_similarities

    def plot_heatmap(self, data: np.ndarray,
                     cmap: str = 'viridis',
                     title: str = 'Heatmap',
                     annot: bool = False,
                     fmt: str = ".2f",
                     figsize: tuple = (8, 6),
                     cbar: bool = True):

        plt.figure(figsize=figsize)
        sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, cbar=cbar)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def is_update_key_sample(self, pre_activations, num_classes):
        if len(self.clas_use_index) == 0:
            return True

        key_class_scores = torch.zeros((num_classes, num_classes)).cuda()
        noise_scores = torch.zeros((len(pre_activations), num_classes)).cuda()
        for i in range(num_classes):
            class_i_index = self.clas_use_index[i]

            pre_distances = self.cosine_similarities(pre_activations, class_i_index, class_j_index=None)
            for j in range(num_classes):
                class_j_index = self.clas_use_index[j]
                mean_class_j_distance = pre_distances[:, class_j_index].mean(dim=1)
                noise_scores[class_i_index, j] = mean_class_j_distance

            noise_scores[class_i_index, :] = self.row_normalize(noise_scores[class_i_index, :])
            key_class_scores[i, :] = noise_scores[class_i_index, :].mean(dim=0)

        sim_scores = torch.diag(key_class_scores).sum()
        off_diag_mask = ~torch.eye(key_class_scores.size(0), dtype=torch.bool, device=key_class_scores.device)
        off_diag_scores = key_class_scores[off_diag_mask].sum()

        if sim_scores > self.best_sim_scores and off_diag_scores < self.worse_off_sim_scores:
            self.best_sim_scores = sim_scores
            self.worse_off_sim_scores = off_diag_scores
            return True
        else:
            return False


    def init_key_sample(self, pre_activations, train_targets, clean_labels_all, num_classes, use_key_selection=True):
        noise_scores = torch.zeros((len(pre_activations), num_classes)).cuda()
        key_class_scores = torch.zeros((num_classes, num_classes)).cuda()
        delete_rate = 0.1 if use_key_selection else 1 - self.class_rate

        self.clas_use_index = [torch.nonzero(train_targets == i, as_tuple=True)[0] for i in range(num_classes)]
        copy_class_indices = [len(indices) for indices in self.clas_use_index]
        while True:
            key_samples_num = len(torch.cat(self.clas_use_index))
            for i in range(num_classes):
                class_i_index = self.clas_use_index[i]

                if len(class_i_index) <= int(copy_class_indices[i] * self.class_rate) or len(class_i_index) <= 2:
                    key_class_scores[i, :] = noise_scores[class_i_index, :].mean(dim=0)
                    continue

                step = max(int(len(class_i_index) * delete_rate), 1)  # 每次去掉 10% 的样本
                pre_distances = self.cosine_similarities(pre_activations, class_i_index, class_j_index=None)

                for j in range(num_classes):
                    class_j_index = self.clas_use_index[j]
                    mean_class_j_distance = pre_distances[:, class_j_index].mean(dim=1)
                    noise_scores[class_i_index, j] = mean_class_j_distance

                noise_scores[class_i_index, :] = self.row_normalize(noise_scores[class_i_index, :])

                scores = noise_scores[class_i_index, i]
                indices_to_remove = torch.argsort(scores)[:step]
                mask = torch.ones(len(class_i_index), dtype=torch.bool, device=class_i_index.device).cuda()
                mask[indices_to_remove] = False
                self.clas_use_index[i] = class_i_index[mask]
                self.opt_pre_activations[class_i_index, :] = pre_activations[class_i_index, :]

                key_class_scores[i, :] = noise_scores[class_i_index, :].mean(dim=0)

            # 检查是否达到 class_rate
            if len(torch.cat(self.clas_use_index)) == key_samples_num:
                break

        self.plot_heatmap(key_class_scores.detach().cpu().numpy(), title='Key Class Scores')

        # sim_scores = torch.diag(key_class_scores).sum()
        # off_diag_mask = ~torch.eye(key_class_scores.size(0), dtype=torch.bool, device=key_class_scores.device)
        # off_diag_scores = key_class_scores[off_diag_mask].sum()
        # if sim_scores > self.best_sim_scores and off_diag_scores < self.worse_off_sim_scores:
        #     self.best_sim_scores = sim_scores
        #     self.worse_off_sim_scores = off_diag_scores

    def attain_second_max(self, x):
        b, c = x.shape  # x: [batch, channels]

        mask = torch.eye(c, dtype=torch.bool, device=x.device)  # [c, c]
        mask = mask.unsqueeze(0).expand(b, c, c)  # [b, c, c]

        x_expanded = x.unsqueeze(1).expand(b, c, c)  # [b, c, c]
        x_masked = x_expanded.masked_fill(mask, float('-inf'))

        second_max_values = x_masked.max(dim=2).values  # [b, c]

        return second_max_values

    def reverse_row_normalize(self, X):
        X_min = X.min(dim=1, keepdim=True).values  # 按行取最小值
        X_max = X.max(dim=1, keepdim=True).values  # 按行取最大值

        denom = X_max - X_min
        denom[denom == 0] = 1.0

        normalized = 1.0 - (X - X_min) / denom
        return normalized

    def compute_doubble_similarity(self, args, pre_activations, train_targets, clean_labels_all, num_classes, use_second_value=True):
        key_class_index = torch.cat(self.clas_use_index).cuda()
        noise_scores = torch.zeros((len(pre_activations), num_classes)).cuda()
        final_cos_noise_scores = torch.zeros(len(pre_activations)).cuda()

        for i in range(num_classes):
            class_i_index = torch.nonzero(train_targets == i, as_tuple=True)[0]

            pre_distances = self.cosine_similarities(pre_activations, class_i_index, class_j_index=key_class_index)
            for j in range(num_classes):
                key_class_j_index = self.clas_use_index[j]
                class_j_index = (key_class_index.unsqueeze(0) == key_class_j_index.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]

                mean_class_j_distance = pre_distances[:, class_j_index].mean(dim=1)
                noise_scores[class_i_index, j] = mean_class_j_distance

            if args.use_only_intra_class:
                second_max_values = self.attain_second_max(noise_scores[class_i_index, :])
                noise_scores[class_i_index, :] = - args.sv_param * second_max_values
            elif use_second_value:
                second_max_values = self.attain_second_max(noise_scores[class_i_index, :])
                noise_scores[class_i_index, :] = noise_scores[class_i_index, :] - args.sv_param * second_max_values
            final_cos_noise_scores[class_i_index] = noise_scores[class_i_index, i]
            final_cos_noise_scores[class_i_index] = self.row_normalize(final_cos_noise_scores[class_i_index].reshape(1, -1)).reshape(-1)
            noise_scores[class_i_index, :] = self.row_normalize(noise_scores[class_i_index, :])

        featrues_pred = torch.argmax(noise_scores, dim=1)

        # plot_heatmap(noise_scores[:10, :].detach().cpu().numpy())
        # print('noisy labels: ' + str(train_targets[:10]))
        # print('clean labels: ' + str(clean_labels_all[:10]))

        return final_cos_noise_scores, noise_scores, featrues_pred


