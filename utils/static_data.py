import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

from utils.utils import write_matrix_to_excel


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)
    sim_weight = torch.ones_like(sim_weight)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # print(pred_scores.shape)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels

def weighted_knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global'):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    return score  # , pred

def divide_knn(feature_bank, labels, num_classes, ids=None):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, num_classes, 200, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max

    return right_score

def reverse_row_normalize(X: np.ndarray) -> np.ndarray:
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    # 避免除以 0
    denom = np.where(X_max - X_min == 0, 1, X_max - X_min)
    normalized = 1 - (X - X_min) / denom
    return normalized

def row_normalize(X: np.ndarray) -> np.ndarray:
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    # 避免除以 0
    denom = np.where(X_max - X_min == 0, 1, X_max - X_min)
    normalized = (X - X_min) / denom
    return normalized

def plot_heatmap(data: np.ndarray,
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

def plot_data_heatmap(pre_activations, targets):
    num_classes = np.unique(targets)
    pre_activations_scores = np.zeros([num_classes.size, num_classes.size])

    for i in range(num_classes.size):
        class_i = np.where(targets == num_classes[i])[0]
        pre_knn = NearestNeighbors(n_neighbors=len(class_i), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i, :].reshape(len(class_i), -1))

        for j in range(num_classes.size):
            class_j = np.where(targets == num_classes[j])[0]

            pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[class_j, :].reshape(len(class_j), -1))
            pre_activations_scores[i, j] = pre_distances.mean()

    pre_activations_scores = reverse_row_normalize(pre_activations_scores)
    plot_heatmap(pre_activations_scores)

    print()

def compute_knn_heatmap(pre_activations, targets):
    num_classes = np.unique(targets)
    pre_activations_scores = np.zeros([num_classes.size, num_classes.size])

    for i in range(num_classes.size):
        class_i = np.where(targets == num_classes[i])[0]
        pre_knn = NearestNeighbors(n_neighbors=len(class_i), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i, :].reshape(len(class_i), -1))

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])

        for j in range(num_classes.size):
            class_j = np.where(targets == num_classes[j])[0]
            pre_activations_scores[i, j] = pre_distances[class_j, :].mean()

    pre_activations_scores = reverse_row_normalize(pre_activations_scores)
    plot_heatmap(pre_activations_scores)

    print()

def compute_class_knn_heatmap(pre_activations, targets, class_rate=0.3):
    num_classes = np.unique(targets)

    all_distances = np.zeros((len(pre_activations)))
    class_index_list = []
    for i in range(num_classes.size):
        class_i = np.where(targets == num_classes[i])[0]
        pre_knn = NearestNeighbors(n_neighbors=len(class_i), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i, :].reshape(len(class_i), -1))

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[class_i, :])
        mean_class_i_distance = reverse_row_normalize(pre_distances.mean(axis=1).reshape(1, -1)).reshape(-1)
        all_distances[class_i] = mean_class_i_distance

        indices = np.argsort(mean_class_i_distance)[int(len(class_i) * (1 - class_rate)):]
        class_index_list.append(class_i[indices.tolist()])
        print()

    print()

def flip_vector(x: np.ndarray) -> np.ndarray:
    return x.max() + x.min() - x

def compute_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    targets[0] = 0

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            # rev_distances = reverse_row_normalize(pre_distances[:, class_j_index])
            # mean_class_j_distance = rev_distances.mean(axis=1)
            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])

        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(class_scores)

    plot_heatmap(noise_scores[:30, :])
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()

def our_knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores.detach().cpu().numpy()

def test_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    targets[0] = 0

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            temp_dist_for_mean = np.copy(pre_distances[:, class_j_index])
            query_indices_in_j_that_are_also_in_i = np.where(np.isin(class_j_index, class_i_index))[0]
            if len(query_indices_in_j_that_are_also_in_i) > 0:
                temp_dist_for_mean[0, query_indices_in_j_that_are_also_in_i] = np.nan
            mean_class_j_distance = np.nanmean(temp_dist_for_mean, axis=1)

            # mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])

        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(class_scores)

    # plot_heatmap(key_class_scores)

    plot_heatmap(noise_scores[:30, :])
    # print('class key index: ' + str(class_index_list[0]))
    # print('class_0_index: ' + str(class_0_index))
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()

def compute_all_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    targets[0] = 0
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = our_knn_predict(pre_activations[class_i_index.tolist(), :], pre_activations.T, torch.arange(len(pre_activations)).cuda(),
                                        len(pre_activations), len(pre_activations))
        # pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])

        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(class_scores)

    # plot_heatmap(key_class_scores)

    plot_heatmap(noise_scores[:30, :])
    # print('class key index: ' + str(class_index_list[0]))
    # print('class_0_index: ' + str(class_0_index))
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()


def compute_class_heatmap(pre_activations, targets, clean_train_target, class_rate=0.3):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
            pre_knn.fit(pre_activations[class_i_index, :])

            pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[class_j_index, :])
            pre_distances = pre_distances.transpose()

            mean_class_j_distance = pre_distances[:, :].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])

    plot_heatmap(class_scores)

    # class_0_index = np.where(targets == num_classes[0])[0]
    # print('class key index: ' + str(class_index_list[0]))
    # print('class_0_index: ' + str(class_0_index))
    # print('noisy labels: ' + str(targets[class_0_index]))
    # print('clean labels: ' + str(clean_train_target[class_0_index]))

    plot_heatmap(noise_scores[:30, :])
    print('front 30 noisy labels: ' + str(targets[:30]))
    print('front 30 clean labels: ' + str(clean_train_target[:30]))

    print()

def main():
    dataset = np.load('./activations/data.npy')
    targets = np.load('./activations/targets.npy')
    pre_activations = np.load('./activations/pre_activations.npy')
    post_activations = np.load('./activations/post_activations.npy')

    num_classes = np.unique(targets)
    pre_activations_scores = np.zeros([num_classes.size, num_classes.size])
    post_activations_scores = np.zeros([num_classes.size, num_classes.size])

    for i in range(num_classes.size):
        class_i = np.where(targets == num_classes[i])[0]
        pre_knn = NearestNeighbors(n_neighbors=len(class_i), algorithm='auto', metric='euclidean')
        post_knn = NearestNeighbors(n_neighbors=len(class_i), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i, :, :].reshape(len(class_i), -1))
        post_knn.fit(post_activations[class_i, :, :].reshape(len(class_i), -1))

        for j in range(num_classes.size):
            class_j = np.where(targets == num_classes[j])[0]

            pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[class_j, :, :].reshape(len(class_j), -1))
            post_distances, post_indices = post_knn.kneighbors(post_activations[class_j, :, :].reshape(len(class_j), -1))
            pre_activations_scores[i, j] = pre_distances.mean()
            post_activations_scores[i, j] = post_distances.mean()

    pre_activations_scores = reverse_row_normalize(pre_activations_scores)
    post_activations_scores = reverse_row_normalize(post_activations_scores)

    plot_heatmap(pre_activations_scores)
    plot_heatmap(post_activations_scores)

    print()

def compute_key_samples_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])
        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(key_class_scores)

    all_key_class_index = np.concatenate(class_index_list).tolist()

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[all_key_class_index, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = np.where(np.isin(all_key_class_index, key_class_j_index))[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)

def compute_key_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.6):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)
        key_class_scores[i, :] = noise_scores[class_index, :].mean(axis=0)

    plot_heatmap(key_class_scores)

    all_key_class_index = np.concatenate(class_index_list).tolist()

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[all_key_class_index, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(all_key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)

    for i in range(1, 2):
        if i * 30 >= len(pre_activations):
            break

        plot_heatmap(noise_scores[(i - 1) * 30: i * 30, :])
        print('sample at %d to %d' % (i, i * 30))
        print('noisy labels: ' + str(targets[(i - 1) * 30: i * 30]))
        print('clean labels: ' + str(clean_train_target[(i - 1) * 30: i * 30]))

    print()

def compute_wright_key_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])
        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(key_class_scores)

    all_key_class_index = np.concatenate(class_index_list).tolist()

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[all_key_class_index, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(all_key_class_index == k)[0][0] for k in key_class_j_index]

            class_j_weigth = reverse_row_normalize(pre_distances[:, class_j_index])
            t = pre_distances[:, class_j_index]
            mean_class_j_distance = np.sum(class_j_weigth * pre_distances[:, class_j_index], axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(noise_scores[:30, :])
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()

def compute_ak_samples_knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.3):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])
        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(key_class_scores)

    ak_class_index_list = []
    for i in range(num_classes.size):
        class_i_ind = np.where(targets == num_classes[i])[0]
        class_i_index = class_index_list[i]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])

        ak_noise_scores = pre_distances.mean(axis=1)
        indices = np.argsort(ak_noise_scores)[:int(len(class_i_ind) * 0.5)]
        ak_class_i_index = list(set(class_i_index) | set(indices.tolist()))
        ak_class_index_list.append(ak_class_i_index)

        key_class_scores[i, :] = noise_scores[ak_class_i_index, :].mean(axis=0)

    plot_heatmap(key_class_scores)

    all_key_class_index = np.concatenate(ak_class_index_list).tolist()

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[all_key_class_index, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(all_key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)

    class_0_index = np.where(targets == num_classes[1])[0]
    plot_heatmap(noise_scores[class_0_index, :])
    print('noisy labels: ' + str(targets[class_0_index]))
    print('clean labels: ' + str(clean_train_target[class_0_index]))

    print()

from cleanlab.internal.constants import EPSILON

def attain_key_samples_weigth(knn_distances):
    # standard_knn_distances = row_normalize(knn_distances)

    scaling_factor = float(max(np.median(knn_distances), 100 * np.finfo(np.float64).eps))
    weighted = np.exp(knn_distances / max(scaling_factor, EPSILON))
    standard_weighted = row_normalize(weighted)

    return standard_weighted

def compute_weight_key_samples_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            class_j_weigth = attain_key_samples_weigth(pre_distances[:, class_j_index])
            mean_class_j_distance = np.sum(class_j_weigth * pre_distances[:, class_j_index], axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])
        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(key_class_scores)

    all_key_class_index = np.concatenate(class_index_list).tolist()

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[all_key_class_index, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(all_key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(noise_scores[:30, :])
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    # for i in range(num_classes.size):
    #     class_i_index = np.where(targets == num_classes[i])[0]
    #     plot_heatmap(noise_scores[class_i_index, :])
    #     print('class %d' % i)
    #     print('noisy labels: ' + str(targets[class_i_index]))
    #     print('clean labels: ' + str(clean_train_target[class_i_index]))

    print()

def test_data(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    targets[0] = 0

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        if i == 0:
            pre_distances_class_0 = pre_distances[:, np.where(targets == 0)[0]]
            pre_distances_class_5 = pre_distances[:, np.where(targets == 5)[0]]
            pre_distances_class_0_5 = np.concatenate((pre_distances_class_0, pre_distances_class_5), axis=1)

            pre_distances_class_0_5 = np.round(pre_distances_class_0_5, 2)
            write_matrix_to_excel(pre_distances_class_0_5, np.arange(pre_distances_class_0_5.shape[1]).astype(str), file_name='pre_distances_class_0_5.xls')

        # if i == 5:
        #     pre_distances_class_5 = pre_distances[:, np.where(targets == 5)[0]]
        #     pre_distances_class_0 = pre_distances[:, np.where(targets == 0)[0]]
        #     pre_distances_class_5_0 = np.concatenate((pre_distances_class_5, pre_distances_class_0), axis=1)
        #
        #     pre_distances_class_5_0 = np.round(pre_distances_class_5_0, 2)
        #     write_matrix_to_excel(pre_distances_class_5_0, np.arange(pre_distances_class_5_0.shape[1]).astype(str), file_name='pre_distances_class_5_0.xls')

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            # sub_array = pre_distances[:, class_j_index]  # shape: (n_samples, k)
            # def row_mean_exclude_top20(row):
            #     thr = np.quantile(row, 0.8)  # 80% 分位数
            #     return row[row <= thr].mean()
            # row_means = np.apply_along_axis(row_mean_exclude_top20, 1, sub_array)

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)
        key_class_scores[i, :] = noise_scores[class_index, :].mean(axis=0)
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(key_class_scores)

    for i in range(1, 2):
        if i * 30 >= len(pre_activations):
            break

        plot_heatmap(noise_scores[(i - 1) * 30: i * 30, :])
        print('sample at %d to %d' % (i, i * 30))
        print('noisy labels: ' + str(targets[(i - 1) * 30: i * 30]))
        print('clean labels: ' + str(clean_train_target[(i - 1) * 30: i * 30]))

    print()


def cosine_similarities(pre_activations, class_i_index, class_j_index=None):
    # 取出目标向量
    vec = pre_activations[class_i_index, :]
    vec_norm = np.linalg.norm(vec)

    if class_j_index is None:
        # 与所有向量比较
        dot_products = vec @ pre_activations.T
        all_norms = np.linalg.norm(pre_activations, axis=1)
        cos_similarities = dot_products / (vec_norm * all_norms + 1e-10)
    else:
        # 只取指定的向量
        other_vecs = pre_activations[class_j_index, :]
        dot_products = vec @ other_vecs.T
        all_norms = np.linalg.norm(other_vecs, axis=-1)
        cos_similarities = dot_products / (vec_norm * all_norms + 1e-10)

    return cos_similarities

def test_cos_similarity(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    # targets[0] = 0

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        # pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        # pre_knn.fit(pre_activations[class_i_index, :])

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)
        # pre_distances = pre_distances.transpose()

        # if i == 0:
        #     pre_distances_class_0 = pre_distances[:, np.where(targets == 0)[0]]
        #     pre_distances_class_5 = pre_distances[:, np.where(targets == 5)[0]]
        #     pre_distances_class_0_5 = np.concatenate((pre_distances_class_0, pre_distances_class_5), axis=1)
        #
        #     pre_distances_class_0_5 = np.round(pre_distances_class_0_5, 2)
        #     write_matrix_to_excel(pre_distances_class_0_5, np.arange(pre_distances_class_0_5.shape[1]).astype(str), file_name='pre_distances_class_0_5.xls')

        # if i == 5:
        #     pre_distances_class_5 = pre_distances[:, np.where(targets == 5)[0]]
        #     pre_distances_class_0 = pre_distances[:, np.where(targets == 0)[0]]
        #     pre_distances_class_5_0 = np.concatenate((pre_distances_class_5, pre_distances_class_0), axis=1)
        #
        #     pre_distances_class_5_0 = np.round(pre_distances_class_5_0, 2)
        #     write_matrix_to_excel(pre_distances_class_5_0, np.arange(pre_distances_class_5_0.shape[1]).astype(str), file_name='pre_distances_class_5_0.xls')

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)
        key_class_scores[i, :] = noise_scores[class_index, :].mean(axis=0)
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(key_class_scores)

    for i in range(1, 2):
        if i * 30 >= len(pre_activations):
            break

        plot_heatmap(noise_scores[(i - 1) * 30: i * 30, :])
        print('sample at %d to %d' % (i, i * 30))
        print('noisy labels: ' + str(targets[(i - 1) * 30: i * 30]))
        print('clean labels: ' + str(clean_train_target[(i - 1) * 30: i * 30]))

    print()

def key_cos_similarity(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)
        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)
        key_class_scores[i, :] = noise_scores[class_index, :].mean(axis=0)
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(key_class_scores)

    key_class_index = np.concatenate(class_index_list).tolist()
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=key_class_index)
        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)

    for i in range(1, 2):
        if i * 30 >= len(pre_activations):
            break

        plot_heatmap(noise_scores[(i - 1) * 30: i * 30, :])
        print('sample at %d to %d' % (i, i * 30))
        print('noisy labels: ' + str(targets[(i - 1) * 30: i * 30]))
        print('clean labels: ' + str(clean_train_target[(i - 1) * 30: i * 30]))

    print()

def key_double_cos_similarity(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    double_cos_noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)
        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)
        key_class_scores[i, :] = noise_scores[class_index, :].mean(axis=0)
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)
    plot_heatmap(key_class_scores)

    key_class_index = np.concatenate(class_index_list).tolist()
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=key_class_index)
        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        all_mask = np.ones_like(noise_scores[class_i_index, :])
        all_mask[np.arange(all_mask.shape[0]), i] = 0
        noise_scores[class_i_index, i] -= (all_mask * noise_scores[class_i_index, :]).max(axis=1)
        double_cos_noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        # noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = double_cos_noise_scores[class_i_index, :].mean(axis=0)

    plot_heatmap(class_scores)

    for i in range(1, 2):
        if i * 30 >= len(pre_activations):
            break

        plot_heatmap(double_cos_noise_scores[(i - 1) * 30: i * 30, :])
        print('sample at %d to %d' % (i, i * 30))
        print('noisy labels: ' + str(targets[(i - 1) * 30: i * 30]))
        print('clean labels: ' + str(clean_train_target[(i - 1) * 30: i * 30]))

    print()

def compute_double_cos_similarity(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    double_cos_noise_scores = np.zeros(len(pre_activations))

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)
        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index = class_i_index[indices.tolist()]
        class_index_list.append(class_index)

    key_class_index = np.concatenate(class_index_list).tolist()
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=key_class_index)
        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]
            class_j_index = [np.where(key_class_index == k)[0][0] for k in key_class_j_index]

            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        all_mask = np.ones_like(noise_scores[class_i_index, :])
        all_mask[np.arange(all_mask.shape[0]), i] = 0
        double_cos_noise_scores[class_i_index] = noise_scores[class_i_index, i] - (all_mask * noise_scores[class_i_index, :]).max(axis=1)
        double_cos_noise_scores[class_i_index] = row_normalize(double_cos_noise_scores[class_i_index].reshape(1, -1)).reshape(-1)

    return double_cos_noise_scores

def knn_heatmap(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    # targets[38] = 1

    class_index_list = []
    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        # pre_knn = NearestNeighbors(n_neighbors=len(pre_activations), algorithm='auto', metric='euclidean')
        # pre_knn.fit(pre_activations[:, :])
        # pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[class_i_index, :])

        pre_knn = NearestNeighbors(n_neighbors=len(class_i_index), algorithm='auto', metric='euclidean')
        pre_knn.fit(pre_activations[class_i_index, :])
        pre_distances, pre_indices = pre_knn.kneighbors(pre_activations[:, :])
        pre_distances = pre_distances.transpose()

        for j in range(num_classes.size):
            class_j_index = np.where(targets == num_classes[j])[0]

            # if i == j:
            # n = len(class_i_index)
            # mask = np.ones((n, n), dtype=bool)
            # np.fill_diagonal(mask, False)
            # masked_array = pre_distances[:, class_j_index][mask]
            # reshaped_array = masked_array.reshape(n, n - 1)
            # mean_class_j_distance = np.mean(reshaped_array, axis=1)
            # else:
            mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)

            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = reverse_row_normalize(noise_scores[class_i_index, :])
        class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        indices = np.argsort(noise_scores[class_i_index, i])[int(len(class_i_index) * (1 - class_rate)):]
        class_index_list.append(class_i_index[indices.tolist()])

        key_class_scores[i, :] = noise_scores[class_i_index[indices.tolist()], :].mean(axis=0)

    plot_heatmap(class_scores)

    plot_heatmap(noise_scores[:30, :])
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()

def extract_key_cos_similarity(pre_activations, targets, clean_train_target, class_rate=0.2):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    double_cos_noise_scores = np.zeros((len(pre_activations), num_classes.size))
    class_scores = np.zeros((num_classes.size, num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_indices = [np.where(targets == num_classes[i])[0] for i in range(num_classes.size)]
    copy_class_indices = [indices.copy() for indices in class_indices]

    total_steps = int(np.ceil((1 - class_rate) / 0.1))  # 计算需要多少步达到 class_rate

    while True:
        key_samples_num = np.concatenate(class_indices).size
        for i in range(num_classes.size):
            class_i_index = class_indices[i]

            if len(class_i_index) <= int(len(copy_class_indices[i]) * class_rate) or len(class_i_index) <= 2:
                key_class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)
                continue

            step = max(int(len(class_i_index) * 0.1), 1)  # 每次去掉 10% 的样本
            pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)

            for j in range(num_classes.size):
                class_j_index = class_indices[j]
                mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
                noise_scores[class_i_index, j] = mean_class_j_distance

            # 归一化噪声分数
            noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

            # 按照噪声分数排序，去掉分数最低的 10%
            indices_to_remove = np.argsort(noise_scores[class_i_index, i])[:step]
            mask = np.ones(len(class_i_index), dtype=bool)
            mask[indices_to_remove] = False
            class_indices[i] = class_i_index[mask]

            key_class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)
            class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        # plot_heatmap(class_scores)
        plot_heatmap(key_class_scores)

        # 检查是否达到 class_rate
        if np.concatenate(class_indices).size == key_samples_num:
            break

    # plot_heatmap(class_scores)
    # plot_heatmap(key_class_scores)

    print()

def extract_key_samples(pre_activations, targets, clean_train_target, class_rate=0.3):
    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))
    key_class_scores = np.zeros((num_classes.size, num_classes.size))

    class_indices = [np.where(targets == num_classes[i])[0] for i in range(num_classes.size)]
    copy_class_indices = [indices.copy() for indices in class_indices]

    while True:
        key_samples_num = np.concatenate(class_indices).size
        for i in range(num_classes.size):
            class_i_index = class_indices[i]

            if len(class_i_index) <= int(len(copy_class_indices[i]) * class_rate) or len(class_i_index) <= 2:
                key_class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)
                continue

            step = max(int(len(class_i_index) * 0.1), 1)  # 每次去掉 10% 的样本
            pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)

            for j in range(num_classes.size):
                class_j_index = class_indices[j]
                mean_class_j_distance = pre_distances[:, class_j_index].mean(axis=1)
                noise_scores[class_i_index, j] = mean_class_j_distance

            # 归一化噪声分数
            noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

            # 按照噪声分数排序，去掉分数最低的 10%
            indices_to_remove = np.argsort(noise_scores[class_i_index, i])[:step]
            mask = np.ones(len(class_i_index), dtype=bool)
            mask[indices_to_remove] = False
            class_indices[i] = class_i_index[mask]

            key_class_scores[i, :] = noise_scores[class_i_index, :].mean(axis=0)

        # 检查是否达到 class_rate
        if np.concatenate(class_indices).size == key_samples_num:
            break

    plot_heatmap(key_class_scores)

    return class_indices

def compute_noise_scores(pre_activations, targets, clean_train_target, class_rate=0.2):
    class_index_list = extract_key_samples(pre_activations, targets, clean_train_target, class_rate=class_rate)

    num_classes = np.unique(targets)
    noise_scores = np.zeros((len(pre_activations), num_classes.size))

    for i in range(num_classes.size):
        class_i_index = np.where(targets == num_classes[i])[0]

        pre_distances = cosine_similarities(pre_activations, class_i_index, class_j_index=None)
        for j in range(num_classes.size):
            key_class_j_index = class_index_list[j]

            mean_class_j_distance = pre_distances[:, key_class_j_index].mean(axis=1)
            noise_scores[class_i_index, j] = mean_class_j_distance

        noise_scores[class_i_index, :] = row_normalize(noise_scores[class_i_index, :])

    plot_heatmap(noise_scores[:30, :])
    print('noisy labels: ' + str(targets[:30]))
    print('clean labels: ' + str(clean_train_target[:30]))

    print()

if __name__ == '__main__':
    main()

