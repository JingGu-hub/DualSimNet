import numpy as np
import torch
import torch.nn.functional as F

class FeaturesProbKnowledgeDistilling:
    def __init__(self, key_samples=None, features_pred=None, num_classes=None):
        self.key_samples = key_samples
        self.features_pred = features_pred
        self.num_classes = num_classes

    def update_parameters(self, key_samples, features_pred):
        self.key_class_index = torch.cat(key_samples).cuda()
        self.features_pred = features_pred
        self.key_samples = key_samples

    def distance_rda(self, z_tilde, z):
        norm_tilde = z_tilde.norm(p=2, dim=1)  # [N]
        norm_z = z.norm(p=2, dim=1)  # [M]

        diff_all = (norm_tilde.unsqueeze(1) - norm_z.unsqueeze(0)).abs()  # [N, M]
        # mu = diff_all.mean(dim=1)  # [N]
        #
        # res = diff_all / mu.unsqueeze(1)  # [N, M]

        return diff_all

    def compute_distance_rda(self, features_teacher, outputs_all, features_student, samples_index, key_class_index, cur_core_idx, other_core_idx):
        with torch.no_grad():
            t_d = self.distance_rda(features_teacher[samples_index], features_teacher[key_class_index])

        s_d = self.distance_rda(features_student, outputs_all[key_class_index])

        cur_loss = None
        try:
            cur_loss_list, other_loss_list = [], []
            for i, (cur_idx, other_idx) in enumerate(zip(cur_core_idx, other_core_idx)):
                with torch.no_grad():
                    cur_t_d, other_t_d = t_d[:, cur_idx], t_d[:, other_idx]
                    cur_t_d, other_t_d = cur_t_d / (cur_t_d.mean(dim=1).unsqueeze(1) + 1e-10), other_t_d / (other_t_d.mean(dim=1).unsqueeze(1) + 1e-10)

                cur_s_d, other_s_d = s_d[:, cur_idx], s_d[:, other_idx]
                cur_s_d, other_s_d = cur_s_d / (cur_s_d.mean(dim=1).unsqueeze(1) + 1e-10), other_s_d / (other_s_d.mean(dim=1).unsqueeze(1) + 1e-10)

                cur_loss = F.smooth_l1_loss(cur_s_d, cur_t_d, reduction='elementwise_mean')
                # other_loss = F.smooth_l1_loss(other_s_d, other_t_d, reduction='elementwise_mean')
                cur_loss_list.append(cur_loss)
                # other_loss_list.append(other_loss)

            cur_loss = torch.stack(cur_loss_list).mean()
            # other_loss = torch.stack(other_loss_list).mean()
        except Exception as e:
            print(e)

        if cur_loss == None:
            print()

        return cur_loss, 0

    def angle_rda(self, z_tilde, z):
        d = (z_tilde.unsqueeze(0) - z.unsqueeze(1))
        norm_d = F.normalize(d, p=2, dim=2)
        angle = torch.bmm(norm_d, norm_d.transpose(1, 2))

        return angle

    def compute_angle_rda(self, features_teacher, outputs_all, features_student, samples_index, key_class_index, cur_core_idx, other_core_idx):
        with torch.no_grad():
            t_d = self.angle_rda(features_teacher[samples_index], features_teacher[key_class_index])
        s_d = self.angle_rda(features_student, outputs_all[key_class_index])

        cur_loss_sum, other_loss_sum = 0, 0
        cur_loss_num, other_loss_num = 0, 0
        for i, (cur_idx, other_idx) in enumerate(zip(cur_core_idx, other_core_idx)):
            try:
                with torch.no_grad():
                    cur_t_d, other_t_d = t_d[cur_idx, :, :], t_d[other_idx, :, :]
                cur_s_d, other_s_d = s_d[cur_idx, :, :], s_d[other_idx, :, :]

                cur_loss = F.smooth_l1_loss(cur_s_d, cur_t_d, reduction='elementwise_mean')
                # other_loss = F.smooth_l1_loss(other_s_d, other_t_d, reduction='elementwise_mean')
                cur_loss_sum += cur_loss * len(cur_idx)
                # other_loss_sum += other_loss * len(other_idx)
                cur_loss_num += len(cur_idx)
                # other_loss_num += len(other_idx)
            except Exception as e:
                print(e)

        cur_loss = cur_loss_sum / cur_loss_num
        # other_loss = other_loss_sum / other_loss_num

        return cur_loss, 0

    def cosine_similarities(self, pre_activations, class_i_index, class_key_index):
        vec = pre_activations[class_i_index, :]  # shape: (d,)
        vec_norm = torch.norm(vec, p=2)  # ||vec||

        key_vecs = pre_activations[class_key_index, :]  # shape: (k, d)
        dot_products = torch.matmul(vec, key_vecs.T)  # shape: (k,)
        all_norms = torch.norm(key_vecs, p=2, dim=1)  # ||each row||
        cos_similarities = dot_products / (vec_norm * all_norms + 1e-10)

        return cos_similarities

    def info_nce_loss(self, anchors, positives, negatives, temperature=0.07):
        anchors = F.normalize(anchors, dim=-1)  # (B, d)
        positives = F.normalize(positives, dim=-1)  # (B, P, d)
        negatives = F.normalize(negatives, dim=-1)  # (B, K, d)

        pos_sim = torch.einsum('bd,bpd->bp', anchors, positives) / temperature # (B, P)
        neg_sim = torch.einsum('bd,bkd->bk', anchors, negatives) / temperature # (B, K)
        pos_exp = torch.exp(pos_sim)  # (B, P)
        neg_exp = torch.exp(neg_sim)  # (B, K)

        numerator = pos_exp.sum(dim=1)  # (B,)
        denominator = pos_exp.sum(dim=1) + neg_exp.sum(dim=1)  # (B,)
        loss = -torch.log(numerator / denominator + 1e-10)  # Loss = -log(numerator / denominator)

        return loss.mean()

    def compute_contra_loss(self, features_teacher, samples_index, pos_core_idx, neg_core_idx):
        pos_pairs, neg_pairs = torch.stack(pos_core_idx), torch.stack(neg_core_idx)

        anchors_teacher = features_teacher[samples_index]
        pos_pairs_teacher = features_teacher[pos_pairs]
        neg_pairs_student = features_teacher[neg_pairs]

        ct_loss = self.info_nce_loss(anchors_teacher, pos_pairs_teacher, neg_pairs_student)

        return ct_loss

    def compute_rda(self, features_teacher, outputs_all, features_student, samples_index):
        cur_core_idx = [
            torch.nonzero(torch.isin(self.key_class_index, self.key_samples[self.features_pred[i]]), as_tuple=True)[0]
            for i in samples_index
        ]
        other_core_idx = [
            torch.nonzero(~torch.isin(self.key_class_index, self.key_samples[self.features_pred[i]]), as_tuple=True)[0]
            for i in samples_index
        ]

        dist_cur_loss, dist_other_loss = self.compute_distance_rda(features_teacher, outputs_all, features_student, samples_index, self.key_class_index, cur_core_idx, other_core_idx)
        angle_cur_loss, angle_other_loss = self.compute_angle_rda(features_teacher, outputs_all, features_student, samples_index, self.key_class_index, cur_core_idx, other_core_idx)

        # rda_loss = (dist_cur_loss + dist_other_loss + angle_cur_loss + angle_other_loss) / 4
        cur_loss = (dist_cur_loss + angle_cur_loss) / 2
        # other_loss = (dist_other_loss + angle_other_loss) / 2

        return cur_loss, 0
