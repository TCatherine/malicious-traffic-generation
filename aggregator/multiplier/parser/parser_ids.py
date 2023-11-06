import pandas
import torch
import sys
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def choose_color(l):
    if l == 'BENIGN':
        return 'yellow'
    if l == 'Web Attack � Sql Injection':
        return 'red'
    if l == 'Web Attack � XSS':
        return 'blue'
    if l == 'Web Attack � Brute Force':
        return 'green'

def get_features_data_explain(data, label):
    pca = PCA(n_components=3)

    # Fit and transform data
    pca.fit_transform(data)

    # Bar plot of explained_variance
    plt.bar(
        range(1, len(pca.explained_variance_) + 1),
        pca.explained_variance_
    )

    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.show()


def show_pca(data, label):
    get_features_data_explain(data, label)
    pca = PCA(n_components=2)

    labels = pandas.unique(label)
    new_data = []
    targets = []
    for l in labels:
        idx = label.index[label == l].tolist()
        new_data.append(data.loc[idx[:100]])
        targets.append(label.loc[idx[:100]])
    df = new_data[0]
    target = targets[0]

    for d, t, in zip(new_data[1:], targets[1:]):
        df = pandas.concat([df, d], ignore_index=True)
        target = pandas.concat([target, t], ignore_index=True)
        df.reset_index()
        target.reset_index()

    components = pca.fit_transform(df)
    plt.figure(figsize=(8, 6))
    colors = [choose_color(l) for l in target]

    plt.scatter(components[:, 0], components[:, 1], c=colors, cmap='rainbow')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


cic_ids2017_dataset = 'data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'


def processing(data, label):
    l = pandas.unique(label)[1]
    idx = label.index[label == l].tolist()
    new_data = data.loc[idx]


def filter_data(data, label):
    labels = [(pandas.unique(label)[2])]
    new_data = []
    targets = []
    for l in labels:
        idx = label.index[label == l].tolist()
        new_data.append(data.loc[idx[:1000]])
        targets.append(label.loc[idx[:1000]])
    df = new_data[0]
    target = targets[0]

    for d, t, in zip(new_data[1:], targets[1:]):
        df = pandas.concat([df, d], ignore_index=True)
        target = pandas.concat([target, t], ignore_index=True)
        df.reset_index()
        target.reset_index()

    return df, target


def parse(
        batch_size,
        file_name=cic_ids2017_dataset,
        is_cuda=False
) -> DataLoader:
    data = pandas.read_csv(
        file_name,
        sep=',',
        skipinitialspace=True)

    features = data.drop('Destination Port', axis=1)
    features = features.drop('Flow Bytes/s', axis=1)
    features = features.drop('Flow Packets/s', axis=1)

    # features["Flow Bytes/s"][features["Flow Bytes/s"] == np.inf] = sys.float_info.max
    # features["Flow Packets/s"][features["Flow Packets/s"] == np.inf] = sys.float_info.max
    # features = features.dropna(subset="Flow Bytes/s")
    # features = features.dropna(subset="Flow Packets/s")

    label = features['Label']
    le = preprocessing.LabelEncoder()

    features = features.drop('Label', axis=1)

    features, label = filter_data(features, label)

    features_values = features.values
    targets = le.fit_transform(label)
    train_target = torch.as_tensor(targets)

    # show_pca(features, label)

    # features = (features - features.mean()) / (features.std() + 0.0001)
    train = torch.tensor(features_values)
    train = (train - train.mean(axis=0, keepdim=True)) / (train.std(axis=0, keepdim=True) + 0.0001)
    train_tensor = TensorDataset(train, train_target)
    res = DataLoader(
        dataset=train_tensor,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True)

    if is_cuda:
        train.to(torch.device("cuda:0"))  # put data into GPU entirely
        train_target.to(torch.device("cuda:0"))
    return res
