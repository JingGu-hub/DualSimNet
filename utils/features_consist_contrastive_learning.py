import torch
import torch.nn.functional as F

class FCCLearning:
    def __init__(self, pre_activations=None, key_samples=None, features_pred=None):
        self.pre_activations = pre_activations
        self.key_samples = key_samples
        self.features_pred = features_pred
        self.key_class_index = None

        self.use_tc_loss = False

    def update_parameters(self, pre_activations=None, key_samples=None, features_pred=None):
        self.pre_activations = pre_activations if pre_activations is not None else self.pre_activations
        self.key_class_index = torch.cat(key_samples).cuda() if key_samples is not None else self.key_class_index
        self.features_pred = features_pred if features_pred is not None else self.features_pred
        self.key_samples = key_samples if key_samples is not None else self.key_samples

        self.use_tc_loss = (self.pre_activations is not None and self.key_class_index is not None and self.features_pred is not None and self.key_samples is not None)

    def pad_sequences(self, sequences, pad_value=-1):
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool).cuda()

        for i, seq in enumerate(sequences):
            length = len(seq)
            padded[i, :length] = torch.tensor(seq, dtype=torch.long)
            mask[i, :length] = 1
        return padded, mask

    def info_nce_loss(self, anchors, positives, negatives, pos_mask, neg_mask, temperature=0.07):
        anchors = F.normalize(anchors, dim=-1)  # (B, d)
        positives = F.normalize(positives, dim=-1)  # (B, P, d)
        negatives = F.normalize(negatives, dim=-1)  # (B, K, d)

        pos_sim = torch.einsum('bd,bpd->bp', anchors, positives) / temperature # (B, P)
        neg_sim = torch.einsum('bd,bkd->bk', anchors, negatives) / temperature # (B, K)
        pos_exp = torch.exp(pos_sim) * pos_mask.float()  # (B, P)
        neg_exp = torch.exp(neg_sim) * neg_mask.float()  # (B, K)

        numerator = pos_exp.sum(dim=1)  # (B,)
        denominator = pos_exp.sum(dim=1) + neg_exp.sum(dim=1)  # (B,)
        loss = -torch.log(numerator / denominator + 1e-10)  # Loss = -log(numerator / denominator)

        return loss.mean()

    def compute_contra_loss(self, features, samples_index, pos_core_idx, neg_core_idx):
        anchors_teacher = features                            # (B,d)
        pos_idx, pos_mask = self.pad_sequences(pos_core_idx)  # (B,Pmax), (B,Pmax)
        neg_idx, neg_mask = self.pad_sequences(neg_core_idx)  # (B,Kmax), (B,Kmax)

        positives = self.pre_activations[pos_idx]  # (B,Pmax,d)
        negatives = self.pre_activations[neg_idx]  # (B,Kmax,d)

        ct_loss = self.info_nce_loss(anchors_teacher, positives, negatives, pos_mask, neg_mask)

        return ct_loss

    def compute_contrastive_loss(self, features, samples_index):
        if not self.use_tc_loss:
            return 0.0

        cur_core_idx = [
            torch.nonzero(torch.isin(self.key_class_index, self.key_samples[self.features_pred[i]]), as_tuple=True)[0]
            for i in samples_index
        ]
        other_core_idx = [
            torch.nonzero(~torch.isin(self.key_class_index, self.key_samples[self.features_pred[i]]), as_tuple=True)[0]
            for i in samples_index
        ]

        ct_loss = self.compute_contra_loss(features, samples_index, cur_core_idx, other_core_idx)

        return ct_loss

