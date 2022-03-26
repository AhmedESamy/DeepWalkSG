import torch
import pickle
import random
import networkx
import numpy as np
import operator
import functools

from tqdm import tqdm
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, dataObject, args):
        self.args = args
        self.targets = dataObject.targets
        self.contexts = dataObject.contexts
        self.negatives = dataObject.negs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.contexts[idx], self.negatives[idx]


class randomwalk(object):
    def __init__(self, graph, args):
        """
        :params graph: NetworkX graph.am args: Arguments object.
        """
        super(randomwalk, self).__init__()
        self.args = args
        self.graph = graph
        self.graph_adj = {}   # {node: {neighbour: edgetype}}
        self.noise_dist = []
        self.pairs_index = []
        self.walks = []
        self.targets = []#torch.tensor([]).to(dtype=torch.long, device=self.args.device)
        self.contexts = [] #torch.tensor([]).to(dtype=torch.long, device=self.args.device)
        self.negs = []
        self.initialize()


    def initialize(self):
        self.graph_adj = networkx.to_dict_of_dicts(self.graph)
        self.noise_dist = torch.tensor([n for n in self.graph.nodes() for _ in range(int(1 + self.graph.degree(n) ** 0.75))])
        for j in range(self.args.walk_length):
            for k in range(j + 1, min(self.args.walk_length, (j + 1) + self.args.window_size)):
                self.pairs_index.append([j, k])
                self.pairs_index.append([k, j])
        self.pairs_index = torch.tensor(self.pairs_index).to(dtype=torch.long, device=self.args.device)


    def generate_walks(self):
        print("\nGenerating walks...")
        nodes = list(self.graph.nodes())
        for _ in tqdm(range(self.args.number_walks)):
            random.shuffle(nodes)
            for v in nodes:
                self.walks.append(self._do_random_walk(v))
        self.walks = torch.tensor(self.walks)


    def _do_random_walk(self, v):
        walk = [v]
        for i in range(self.args.walk_length-1):
            current_node = walk[-1]
            walk.append(random.choice(list((self.graph_adj[current_node]).keys())))
        return walk


    def process_walks(self):
        print("\nProcessing walks...")
        for walk in tqdm(self.walks):
            pairs = walk[self.pairs_index]
            self.targets += pairs[:, 0].tolist()
            self.contexts += pairs[:, 1].tolist()
        self.targets = torch.tensor(self.targets)
        self.contexts = torch.tensor(self.contexts)
        self.negs = self.draw_negatives(self.targets)



    def draw_negatives(self, sources):
        indices = np.random.randint(0, self.graph.number_of_nodes(), size=(len(sources), self.args.negative_samples))
        return self.noise_dist[torch.tensor(indices)]




    '''
    print("\nsaving RW data object...")
    filehandler = open('RWDataObject_' + self.args.dataset + '.obj', 'wb')
    pickle.dump(self, filehandler)
    '''
