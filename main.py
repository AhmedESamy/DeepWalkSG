import torch
import numpy as np

from utils.graph import graph_reader
from utils.parameters.param_parser import parameter_parser
from utils.parameters.dataset_params import dataset_params
from utils.evaluation.node_classification import node_classification
from model.skipgramV2 import word2vec


def main():
    args = parameter_parser()

    args.dataset = "DBLP"  # ACM, DBLP, Foursquare
    args.train = 1  # 0:test, 1:train
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")

    graph = graph_reader(args.dataset, relabel=args.train)

    if args.train ==1:
        w2v = word2vec(graph, args, filename= args.embedding_output_path+'/'+args.dataset+'_w2v')
        w2v.train()

        graph = graph_reader(args.dataset, relabel=0)
        random_seeds = dataset_params[args.dataset]['seeds']
        path_to_labels = dataset_params[args.dataset]['label_path']
        embeddings = np.load(args.embedding_output_path+'/'+args.dataset+'_w2v.npy')
        node_classification(graph, embeddings, path_to_labels, random_seeds, eval=True)



if __name__ == '__main__':
    main()

