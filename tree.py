
import numpy as np


class Tree_node():
    def __init__(self, ids=None, child_node=[], entropy=0, depth=0):
        self.ids = ids  # chi so node chua
        self.child_nodes = []  # danh sach cac node con
        self.entropy = entropy  # entropy muc nhieu loan cua node
        self.depth = depth  # do sau cua node
        self.split_attribute = None  # thuoc tinh de chia child node
        self.split_value = None  # gia tri cu thuoc tinh can chia
        self.label = None  # nhan cua node neu la leaf

    def set_propertie(self, split_attribute, split_value):
        self.split_attribute = split_attribute
        self.split_value = split_value

    def set_label(self, label):
        self.label = label


class Decision_Tree():
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth  # do sau lon nhat cua cay
        self.min_samples_split = min_samples_split  # so luong mau be nhat co the chia
        self.min_gain = min_gain  # do giam be nhat, neu nhor hon khong chia nodfree
        self.n_train = 0  # so luong diem du lieu

    # Tinh entropy
    def _entropy(self, ids):
        if ids.shape[0] == 0:
            return 0
        # tinh tan suat xuat hien cua cac class trong node
        arr, fred = np.unique(self.target[ids], return_counts=True)
        fred = fred / fred.sum()  # tinh xac xuat
        return -np.sum(fred*np.log(fred))  # entropy

    # set label cho diem du lieu
    def _set_label(self, node):
        arr, fred = np.unique(self.target[node.ids], return_counts=True)
        node.set_label(arr[np.argmax(fred)])

    # chia day ids
    def split_ids(self, ids, col, value):
        # ids_node1 = chi so cac phan tu trong node co gia tri cua attribute col <value
        ids_node1 = ids[self.data[ids, col] < value]
        # ids_node2 = chi so cac phan tu trong node co gia tri cua attribute col  >=value
        ids_node2 = ids[self.data[ids, col] >= value]
        return ids_node1, ids_node2

    # chia node thanh child_nodes
    def split(self, node):  # chia node thanh child node
        ids = node.ids
        best_gain = 0  # do giam tot nhat khi chia  attribute
        best_attribute = None  # thuoc tinh tot nhat de chia
        # gia tri cua thuoc tinh(chia theo < value_splits vaf >= value_splits)
        value_split = None
        node_data = self.data[ids, :]  # du lieu cua node
        cols = self.data.shape[1]  # clo laf so thuoc tinh cua du lieu
        best_splits = []  # tap cac chi so chia thanh cac node
        for col in range(cols):
            # tim cac gia tri cua attribute moi gia tri lay 1 lan va sap xep tang dan
            unique_values = np.unique(node_data[:, col])
            for value in unique_values:
                ids_node1, ids_node2 = self.split_ids(ids, col, value)
                splits = np.array([ids_node1, ids_node2])
                p = ids_node1.shape[0]/ids.shape[0]
                gain = node.entropy - p * \
                    self._entropy(ids_node1)-(1-p)*self._entropy(ids_node2)
                if gain < self.min_gain:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = col
                    best_splits = splits
                    value_split = value
        node.set_propertie(best_attribute, value_split)
        child_nodes = [Tree_node(ids=split, entropy=self._entropy(
            split), depth=node.depth+1) for split in best_splits]
        return child_nodes

    # Tim decision tree

    def fit(self, data, target):
        self.n_train = data.shape[0]
        self.data = data
        self.target = target
        ids = np.array(range(self.n_train))
        self.root = Tree_node(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.child_nodes = self.split(node)
                if not node.child_nodes:
                    self._set_label(node)
                queue += node.child_nodes
            else:
                self._set_label(node)

    # du doan label

    def predict(self, data):
        n_data = data.shape[0]
        labels = [None]*n_data
        for i in range(n_data):
            x = data[i, :]
            node = self.root
            while node.child_nodes:
                value = x[node.split_attribute]
                if value < node.split_value:
                    node = node.child_nodes[0]
                else:
                    node = node.child_nodes[1]
            labels[i] = node.label
        return labels
        
