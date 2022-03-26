import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch.optim import SGD
from torch.utils.data import DataLoader
from utils.data.dataObj import randomwalk, Dataset


class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse= True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse= True)
        self.init_emb()

    def init_emb(self):
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.v_embeddings.weight)

    def forward(self, u_pos, v_pos, v_neg):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)
        neg_embed_v = self.v_embeddings(v_neg)

        pos_logits = (embed_u * embed_v).sum(-1) # batch
        pos_loss = torch.nn.functional.logsigmoid(pos_logits)
        neg_logits = torch.bmm(neg_embed_v, embed_u[:, :, None]).squeeze(-1)
        neg_loss = torch.nn.functional.logsigmoid(-neg_logits).sum(-1)
        return -(pos_loss + neg_loss)

    def get_embeddings(self):
            return self.u_embeddings.weight.data.cpu().numpy()

    def save_embedding(self, file_name):
            np.save(file_name, self.get_embeddings())


class word2vec:
    def __init__(self, graph, args, filename=None):
        super(word2vec, self).__init__()
        self.graph = graph
        self.args = args
        self.filename = filename

    def train(self):
        print(self.args.device)
        print("\n**word2vec**")

        path_to_file = 'random_walks_' + self.args.dataset + '.obj'
        if not Path(path_to_file).is_file():
            RandomWalker = randomwalk(self.graph, self.args)
            RandomWalker.generate_walks()
            RandomWalker.process_walks()
        else:
            RandomWalker = pickle.load(open(path_to_file, 'rb'))


        dataset = Dataset(RandomWalker, self.args)
        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            num_workers=self.args.workers, shuffle=True, pin_memory=True)

        sg_model = skipgram(RandomWalker.graph.number_of_nodes(), self.args.dimension)
        sg_model.to(self.args.device)
        min_alpha = 0.0001
        optimizer = torch.optim.SparseAdam(list(sg_model.parameters()), self.args.learning_rate)
        lr_delta = (self.args.learning_rate - min_alpha) / (len(loader) * self.args.iterations)

        print("\nTraining...")
        for _ in (range(self.args.iterations)):
            for batch in tqdm(loader):
                optimizer.zero_grad()
                loss = sg_model(batch[0].to(self.args.device),
                                batch[1].to(self.args.device),
                                batch[2].to(self.args.device))

                loss.mean().backward()
                optimizer.step()
                optimizer.param_groups[0]['lr'] -= lr_delta


        print('\nFinished Training the Deep Walker Model')

        embeddings = sg_model.get_embeddings()
        if self.filename:
            sg_model.save_embedding(self.filename)
        return embeddings



