from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.CINA.embedding_model import CINA_MODEL, HCINA_MODEL, StableFactor, Combine4Model
from utils.graph_utils import load_gt
from algorithms.CINA.utils import *
import torch
import networkx as nx
import random
import numpy as np
from time import time
import dhg
from algorithms.PALE.embedding_model import PaleEmbedding
from algorithms.PALE.mapping_model import PaleMappingLinear
from torch.autograd import Variable
from tqdm import tqdm
import copy


class CINA(NetworkAlignmentModel):
    """
    CINA model for networks alignment task
    """
    def __init__(self, source_dataset, target_dataset, args):
        super(CINA, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.alphas_e = [1, 1, 1]
        self.alphas_h = [1, 1, 1]
        self.args = args
        # full dict is just used in the process of investigating and need to be DELETE later
        self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.gt_train = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    
    def align(self):
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        source_adj = self.source_dataset.get_adjacency_matrix()
        target_adj = self.target_dataset.get_adjacency_matrix()
        new_source_A_hat, new_source_feats, new_deg, new_edges, new_adj, id2idx_augment = self.graph_augmentation(self.source_dataset,target_adj)

        new_adj = Variable(torch.FloatTensor(new_adj), requires_grad = False)
        source_adj = Variable(torch.FloatTensor(source_adj), requires_grad = False)
        target_adj = Variable(torch.FloatTensor(target_adj), requires_grad = False)
        if self.args.cuda:
            new_adj = new_adj.cuda()
            source_adj = source_adj.cuda()
            target_adj = target_adj.cuda()

        new_source_info = {'num_nodes': len(new_source_feats), 'deg': new_deg, 'edges': new_edges, 'adj': new_adj}
        source_info = {'num_nodes': len(source_feats), 'deg': self.source_dataset.get_nodes_degrees(), 'edges': self.source_dataset.get_edges(), 'adj': source_adj}
        target_info = {'num_nodes': len(target_feats), 'deg': self.target_dataset.get_nodes_degrees(), 'edges': self.target_dataset.get_edges(), 'adj': target_adj}

        test_dict = copy.deepcopy(self.full_dict)
        train_dict = copy.deepcopy(self.gt_train)
        self.full_dict = {u:v for (u,v) in id2idx_augment.items() if random.uniform(0,1) > 0.1}
        self.gt_train = {u:v for (u,v) in id2idx_augment.items() if u not in list(self.full_dict.keys())}

        # need compute Ricci curvature and save first utils.compute_Ricci.py
        weight_edges_s, weight_edges_t, weight_edges_t_, weight_edges_s_A, weight_edges_t_A, weight_edges_t_A_ = self.loadCurv()

        self.args.w_mul_s = torch.FloatTensor([i[2] for i in weight_edges_s])
        self.args.w_mul_t = torch.FloatTensor([i[2] for i in weight_edges_t])

        self.args.w_mul_s_A = torch.FloatTensor([i[2] for i in weight_edges_s_A])
        self.args.w_mul_t_A = torch.FloatTensor([i[2] for i in weight_edges_t_A])

        self.args.weight_edges_s = torch.FloatTensor(weight_edges_s)
        self.args.weight_edges_t = torch.FloatTensor(weight_edges_t)

        hg_s = self.constructHG(source_info)
        hg_t = self.constructHG(new_source_info)

        S_CINA_e, S_CINA_h, S_hyper, S_pale = self.get_multi_align(source_A_hat, new_source_A_hat, source_feats, new_source_feats, source_info, new_source_info, hg_s, hg_t)
        combine_model = Combine4Model()
        combine_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, combine_model.parameters()), lr=self.args.lr)

        for epochs in tqdm(range(200)):
            combine_optimizer.zero_grad()
            loss = combine_model.loss(S_CINA_e, S_CINA_h, S_pale, S_hyper.detach().cpu().numpy(), id2idx_augment)
            loss.backward()
            combine_optimizer.step()
            print("Loss: {:.4f}".format(loss.item()))

        self.full_dict = test_dict
        self.gt_train = train_dict

        self.args.w_mul_t = torch.FloatTensor([i[2] for i in weight_edges_t_])
        self.args.weight_edges_t = torch.FloatTensor(weight_edges_t_)
        self.args.w_mul_t_A = torch.FloatTensor([i[2] for i in weight_edges_t_A_])

        hg_t = self.constructHG(target_info)

        S_CINA_e, S_CINA_h, S_hyper, S_pale = self.get_multi_align(source_A_hat, target_A_hat, source_feats, target_feats, source_info, target_info, hg_s, hg_t)
        S = combine_model(S_CINA_e, S_CINA_h, S_pale, S_hyper.detach().cpu().numpy())
        print('-'*100)
        S = S.detach().cpu().numpy()
        return S

    def constructHG(self, info):
        g = dhg.Graph(info['num_nodes'], info['edges'])
        hg = dhg.Hypergraph.from_graph_kHop(g, k=1)
        return hg

    def loadCurv(self):
        weight_edges_s = np.loadtxt(self.args.source_dataset + 'weight.txt')
        weight_edges_t = np.loadtxt(self.args.target_dataset + 'weight.txt')
        weight_edges_t_ = np.loadtxt(self.args.target_dataset + 'weight_.txt')
        weight_edges_s_A = np.loadtxt(self.args.source_dataset + 'weight_A.txt')
        weight_edges_t_A = np.loadtxt(self.args.target_dataset + 'weight_A.txt')
        weight_edges_t_A_ = np.loadtxt(self.args.target_dataset + 'weight_A_.txt')
        return weight_edges_s, weight_edges_t, weight_edges_t_, weight_edges_s_A, weight_edges_t_A, weight_edges_t_A_

    def get_multi_align(self, source_A_hat, target_A_hat, source_feats, target_feats, source_info, target_info, hg_s, hg_t):
        """
        step1: align by GCN
        step2: find stables nodes
        step3: run pale with stabels nodes
        step4: run mincut with stabels nodes


        """
        CINA = CINA_MODEL(
            activate_function='tanh',
            num_GCN_blocks=2,
            input_dim=100,
            output_dim=self.args.embedding_dim,
            num_source_nodes=len(source_A_hat),
            num_target_nodes=len(target_A_hat),
            source_info=source_info,
            target_info=target_info,
            source_feats=source_feats,
            target_feats=target_feats
        )

        HCINA = HCINA_MODEL(
            embedding_dim=self.args.embedding_dim,
            source_feats=source_feats,
            target_feats=target_feats
        )


        if self.args.cuda:
            CINA = CINA.cuda()
            HCINA = HCINA.cuda()

        CINA.train()
        HCINA.train()

        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CINA.parameters()), lr=self.args.lr)
        structural_optimizer_h = torch.optim.Adam(filter(lambda p: p.requires_grad, HCINA.parameters()), lr=self.args.hCINA_lr)

        CINA_S_e, S_CINA_h = self.train_CINA(CINA, source_A_hat, target_A_hat, structural_optimizer, self.args.threshold)
        S_CINA_hyper = self.train_HCINA(HCINA, structural_optimizer_h, hg_s, hg_t)

        gt_train = self.gt_train
        # Step2: PALE
        start = time()
        # self.args.cuda = True
        # source_info['adj'] = source_info['adj'].cuda()
        # target_info['adj'] = target_info['adj'].cuda()

        source_pale, target_pale = self.learn_pale_embs(source_info, target_info)

        pale_map = PaleMappingLinear(
                                    embedding_dim=self.args.embedding_dim,
                                    source_embedding=source_pale,
                                    target_embedding=target_pale,
                                    )

        pale_S = self.map_source_target_pale(pale_map, list(gt_train.keys()), gt_train, source_pale, target_pale)
        print('pale running time: {:.4f}'.format(time() - start))

        return CINA_S_e, S_CINA_h, S_CINA_hyper, pale_S

    def map_source_target_pale(self, pale_map, source_train_nodes, gt_train, source_pale, target_pale):
        """
        source_train_nodes: Numpy array
        gt_train: dictionary
        """
        if self.args.cuda:
            pale_map = pale_map.cuda()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, pale_map.parameters()), lr=self.args.pale_lr)

        pale_map_batchsize = len(source_train_nodes) // 4
        n_iters = len(source_train_nodes) // pale_map_batchsize
        assert n_iters > 0, "batch_size is too large"
        if(len(source_train_nodes) % pale_map_batchsize > 0):
            n_iters += 1
        total_steps = 0
        n_epochs = self.args.pale_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time()
            print('Epochs: ', epoch)
            np.random.shuffle(source_train_nodes)
            for iter in range(n_iters):
                source_batch = source_train_nodes[iter*pale_map_batchsize:(iter+1)*pale_map_batchsize]
                target_batch = [gt_train[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.args.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                loss = pale_map.loss(source_batch, target_batch)
                loss.backward()
                optimizer.step()
            
                total_steps += 1
            print("mapping_loss: {:.4f}".format(loss.item()))
            self.mapping_epoch_time = time() - start

        source_pale_map = pale_map(source_pale)
        self.S_pale = torch.matmul(source_pale_map, target_pale.t())
        self.S_pale = self.S_pale.detach().cpu().numpy()

        return self.S_pale/4


    def learn_pale_embs(self, source_info, target_info):

        num_source_nodes = source_info['num_nodes']
        source_deg = source_info['deg']
        source_edges = source_info['edges']

        num_target_nodes = target_info['num_nodes']
        target_deg = target_info['deg']
        target_edges = target_info['edges']

        #source_edges, target_edges = self.extend_edge(source_edges, target_edges)
        
        print("Done extend edges")
        source_pale = self.learn_pale(num_source_nodes, source_deg, source_edges)
        target_pale = self.learn_pale(num_target_nodes, target_deg, target_edges)
        return source_pale, target_pale


    def learn_pale(self, num_nodes, deg, edges):
        pale_model = PaleEmbedding(
                                    n_nodes = num_nodes,
                                    embedding_dim = self.args.embedding_dim,
                                    deg= deg,
                                    neg_sample_size = 10,
                                    cuda = self.args.cuda,
                                    )
        if self.args.cuda:
            pale_model = pale_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pale_model.parameters()), lr=self.args.pale_lr)
        embedding = self.train_pale_emb(pale_model, edges, optimizer)
        return embedding
    

    def train_pale_emb(self, embedding_model, edges, optimizer):
        num_edges = len(edges)
        n_iters = num_edges // self.args.pale_emb_batchsize
        assert n_iters > 0, "batch_size is too large!"
        if(num_edges % self.args.pale_emb_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.args.pale_epochs
        for epoch in range(1, n_epochs + 1):
            print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            loss, loss0, loss1 = 0, 0, 0
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.args.pale_emb_batchsize:(iter+1)*self.args.pale_emb_batchsize])
                if self.args.cuda:
                    batch_edges = batch_edges.cuda()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                total_steps += 1
            print(
                        "train_loss=", "{:.5f}".format(loss.item()),
                        "true_loss=", "{:.5f}".format(loss0.item()),
                        "neg_loss=", "{:.5f}".format(loss1.item())
                    )
            
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.args.cuda:
            embedding = embedding.cuda()

        return embedding


    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        # Norm2 normalization
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        # features is okey, just A_hat, A_hat is okey too
        return source_A_hat, target_A_hat, source_feats, target_feats


    def graph_augmentation(self, dataset, target_adj, type_aug='remove_edges'):
        """
        Generate small noisy graph from original graph
        :params dataset: original graph
        :params type_aug: type of noise added for generating new graph
        """
        t_nodes = target_adj.shape[0]
        t_edges = int(target_adj.sum())/2

        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()
        padder = t_nodes - adj.shape[0]
        if padder > 0:

            adj = nx.to_numpy_matrix(dataset.G)
            edges = list(dataset.G.edges())
            del_edges = random.sample(list(dataset.G.edges()), int(len(list(dataset.G.edges()))/5))
            #  删边
            for edge in del_edges:
                adj[dataset.id2idx[edge[0]], dataset.id2idx[edge[1]]] = 0
                adj[dataset.id2idx[edge[1]], dataset.id2idx[edge[0]]] = 0
            edges = [(dataset.id2idx[i[0]], dataset.id2idx[i[1]]) for i in edges if i not in del_edges]
            nodes = list(dataset.G.nodes())
            feats = np.copy(dataset.features)
            feats = torch.FloatTensor(feats)
            sidx2idx = {i:i for i in range(len(nodes))}
            deg = adj.sum(axis=1).flatten()
            # H——拉普拉斯变换
            new_adj_H, _ = Laplacian_graph(adj)
            if self.args.cuda:
                feats = feats.cuda()
                new_adj_H = new_adj_H.cuda()
            deg = np.asarray(deg)[0]
            return new_adj_H, feats, deg, edges, adj, sidx2idx
            
        else:
            nodes = [list(dataset.G.nodes())[0]]
            index = 0
            count = 1
            while count < t_nodes:
                try:
                    nodes.extend([i for i in dataset.G.neighbors(nodes[index]) if i not in nodes])
                    count = len(nodes)
                    index += 1
                except:
                    break
            if nx.number_connected_components(dataset.G) > 1:
                nodes = random.sample(list(dataset.G.nodes()), t_nodes)
            new_G = dataset.G.subgraph(nodes)
            edges = list(new_G.edges())
            source_idx = dataset.id2idx
            sidx2idx = {source_idx[node]:i for i,node in enumerate(nodes)} # dict {index of source graph: index of augment graph}
            id2idx = {node:i for i,node in enumerate(nodes)}
            new_G = nx.Graph()
            new_G.add_nodes_from(nodes)
            new_G.add_edges_from(edges)
            adj = nx.to_numpy_matrix(new_G)
            feats = np.copy(dataset.features)
            new_feats = np.zeros((len(nodes), feats.shape[1]))
            s_idx2id = {u:v for v,u in source_idx.items()}
            for i,feat in enumerate(feats):
                if s_idx2id[i] in nodes:
                    new_feats[sidx2idx[i]] = feat
            print(new_feats)
            num_eds = adj.sum()/2 - t_edges
            if num_eds > 0:
                while num_eds > 1:
                    edge = random.choice(edges)
                    adj[id2idx[edge[0]], id2idx[edge[1]]] = 0
                    adj[id2idx[edge[1]], id2idx[edge[0]]] = 0
                    edges.remove(edge)
                    num_eds -= 1
                new_feats = torch.FloatTensor(new_feats)
                deg = adj.sum(axis=1).flatten()
                new_adj_H, _ = Laplacian_graph(adj)
                if self.args.cuda:
                    new_feats = new_feats.cuda()
                    new_adj_H = new_adj_H.cuda()
                deg = np.asarray(deg)[0]
                edges = [(id2idx[e[0]],id2idx[e[1]]) for e in edges]
                print('augment nodes: {}, augment edges: {}'.format(adj.shape[0], adj.sum()/2))
                return new_adj_H, new_feats, deg, edges, adj, sidx2idx
            else:
                stat = 2*num_eds / (adj.shape[0]**2 - adj.sum())
                edges = [(id2idx[e[0]],id2idx[e[1]]) for e in edges]
                for i in range(adj.shape[0]):
                    for j in range(i, adj.shape[0]):
                        if adj[i,j] == 0 and random.uniform(0,1) < stat:
                            adj[i,j] = 1
                            adj[j,i] = 1
                            edges.append((i,j))
                new_feats = torch.FloatTensor(new_feats)
                deg = adj.sum(axis=1).flatten()
                new_adj_H, _ = Laplacian_graph(adj)
                if self.args.cuda:
                    new_feats = new_feats.cuda()
                    new_adj_H = new_adj_H.cuda()
                deg = np.asarray(deg)[0]
                print('augment nodes: {}, augment edges: {}'.format(adj.shape[0], adj.sum()/2))
                return new_adj_H, new_feats, deg, edges, adj, sidx2idx


    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())

        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss
    

    def gcn_semisup_training(self, CINA, source_A_hat, target_A_hat, optimizer):
        train = np.asarray([[x,y] for (x,y) in self.gt_train.items()])

        for epoch in range(self.args.CINA_epochs):
            optimizer.zero_grad()
            print("GAlign learning epoch: {}".format(epoch))

            source_outputs = CINA(source_A_hat, self.args.w_mul_s_A, net='s')
            target_outputs = CINA(target_A_hat, self.args.w_mul_t_A, net='t')
            left = train[:,0]
            right = train[:,1]

            source_last1_output_e = source_outputs[-1][0]
            source_last1_output_h = source_outputs[-1][1]
            target_last1_outputs_e = target_outputs[-1][0]
            target_last1_outputs_h = target_outputs[-1][1]
            left_x1_e = source_last1_output_e[left]
            right_x1_e = target_last1_outputs_e[right]
            left_x1_h = source_last1_output_h[left]
            right_x1_h = target_last1_outputs_h[right]
            sup_loss1_e = (left_x1_e - right_x1_e)**2
            sup_loss1_e = sup_loss1_e.mean()
            sup_loss1_h = (left_x1_h - right_x1_h) ** 2
            sup_loss1_h = sup_loss1_h.mean()
            source_last2_output_e = source_outputs[-2][0]
            source_last2_output_h = source_outputs[-2][1]
            target_last2_outputs_e = target_outputs[-2][0]
            target_last2_outputs_h = target_outputs[-2][1]
            left_x2_e = source_last2_output_e[left]
            right_x2_e = target_last2_outputs_e[right]
            left_x2_h = source_last2_output_h[left]
            right_x2_h = target_last2_outputs_h[right]
            sup_loss2_e = (left_x2_e - right_x2_e) ** 2
            sup_loss2_e = sup_loss2_e.mean()
            sup_loss2_h = (left_x2_h - right_x2_h) ** 2
            sup_loss2_h = sup_loss2_h.mean()
            sup_loss_e = sup_loss1_e + sup_loss2_e
            sup_loss_h = sup_loss1_h + sup_loss2_h
            sup_loss = sup_loss_e + sup_loss_h

            unsup_loss_source_e = self.linkpred_loss(source_outputs[-1][0], source_A_hat) + self.linkpred_loss(
                source_outputs[-2][0], source_A_hat)
            unsup_loss_target_e = self.linkpred_loss(target_outputs[-1][0], target_A_hat) + self.linkpred_loss(
                target_outputs[-2][0], target_A_hat)
            unsup_loss_source_h = self.linkpred_loss(source_outputs[-1][1], source_A_hat) + self.linkpred_loss(
                source_outputs[-2][1], source_A_hat)
            unsup_loss_target_h = self.linkpred_loss(target_outputs[-1][1], target_A_hat) + self.linkpred_loss(
                target_outputs[-2][1], target_A_hat)

            unsup_loss = unsup_loss_source_e + unsup_loss_target_e + unsup_loss_source_h + unsup_loss_target_h

            loss = sup_loss*0.3 + unsup_loss*0.7
            loss.backward()
            print('recent loss: {:.4f}, sup: {}, unsup: {:.4f}'.format(loss, sup_loss, unsup_loss))
            optimizer.step()
        CINA.eval()
        return CINA

    def hyper_gcn_semisup_training(self, HCINA, optimizer, hg_s, hg_t):
        train = np.asarray([[x,y] for (x,y) in self.gt_train.items()])

        for epoch in range(20):
            optimizer.zero_grad()
            print("hyper learning epoch: {}".format(epoch))

            source_outputs = HCINA(HCINA.source_feats, hg_s, v2e_weight=hg_s.v2e_weight, e2v_weight=hg_s.e2v_weight)
            target_outputs = HCINA(HCINA.target_feats, hg_t, v2e_weight=hg_t.v2e_weight, e2v_weight=hg_t.e2v_weight)

            left = train[:,0]
            right = train[:,1]

            sup_loss3 = (source_outputs[left] - target_outputs[right]) ** 2
            sup_loss3 = sup_loss3.mean()
            sup_loss = sup_loss3
            loss = sup_loss
            loss.backward()
            print('recent loss: {:.10f}'.format(loss))
            optimizer.step()
        HCINA.eval()
        return HCINA

    def refine_hyper(self, HCINA, hg_s, hg_t):
        source_outputs = HCINA(HCINA.source_feats, hg_s, v2e_weight=hg_s.v2e_weight, e2v_weight=hg_s.e2v_weight)
        target_outputs = HCINA(HCINA.target_feats, hg_t, v2e_weight=hg_t.v2e_weight, e2v_weight=hg_t.e2v_weight)
        S = torch.matmul(F.normalize(source_outputs), F.normalize(target_outputs).t())
        self.S_hyper = S
        return self.S_hyper


    def refine(self, CINA, source_A_hat, target_A_hat, threshold):
        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)
        if self.args.cuda:
            refinement_model = refinement_model.cuda()
        S_max_e = None
        S_max_h = None
        source_outputs = CINA(refinement_model(source_A_hat, 's'), self.args.w_mul_s_A, 's')
        target_outputs = CINA(refinement_model(target_A_hat, 't'), self.args.w_mul_t_A, 't')
        accs_e, accs_h, S_e, S_h = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas_e, self.alphas_h)
        self.CINA_S_e = S_e
        self.CINA_S_h = S_h

        source_edges = self.source_dataset.get_edges()
        target_edges = self.target_dataset.get_edges()
        edgess = [source_edges.tolist(), target_edges.tolist()]
        score_e = np.mean(S_e.max(axis = 1))
        score_h = np.mean(S_h.max(axis=1))
        acc_max_e = 0
        acc_max_h = 0
        if score_e > refinement_model.score_max_e:
            refinement_model.score_max_e = score_e
            acc_max_e = accs_e
            S_max_e = S_e

        if score_h > refinement_model.score_max_h:
            refinement_model.score_max_h = score_h
            acc_max_h = accs_h
            S_max_h = S_h
        alpha_source_max_e = refinement_model.alpha_source_e + 0
        alpha_source_max_h = refinement_model.alpha_source_h + 0
        alpha_target_max_e = refinement_model.alpha_target_e + 0
        alpha_target_max_h = refinement_model.alpha_target_h + 0
        for epoch in range(self.args.refinement_epochs):
            source_candidates_e, target_candidates_e, len_source_candidates_e, count_true_candidates_e,  source_candidates_h, target_candidates_h, len_source_candidates_h, count_true_candidates_h = self.get_candidate(source_outputs, target_outputs, threshold)

            refinement_model.alpha_source_e[source_candidates_e] *= 1.1
            refinement_model.alpha_target_e[target_candidates_e] *= 1.1
            refinement_model.alpha_source_h[source_candidates_h] *= 1.1
            refinement_model.alpha_target_h[target_candidates_h] *= 1.1

            source_outputs = CINA(refinement_model(source_A_hat, 's'), self.args.w_mul_s_A, 's')
            target_outputs = CINA(refinement_model(target_A_hat, 't'), self.args.w_mul_t_A, 't')
            accs_e, accs_h, S_e, S_h = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas_e, self.alphas_h)
            score_e = np.mean(S_e.max(axis = 1))
            score_h = np.mean(S_h.max(axis=1))
            if score_e > refinement_model.score_max_e:
                refinement_model.score_max_e = score_e
                alpha_source_max_e = refinement_model.alpha_source_e + 0
                alpha_target_max_e = refinement_model.alpha_target_e + 0
                acc_max_e = accs_e
                S_max_e = S_e
            if score_h > refinement_model.score_max_h:
                refinement_model.score_max_h = score_h
                alpha_source_max_h = refinement_model.alpha_source_h + 0
                alpha_target_max_h = refinement_model.alpha_target_h + 0
                acc_max_h = accs_h
                S_max_h = S_h
            if epoch == self.args.refinement_epochs - 1:
                print("Numcandidate_e: {}, num_true_candidate_e: {}".format(len_source_candidates_e, count_true_candidates_e))
                print("Numcandidate_h: {}, num_true_candidate_h: {}".format(len_source_candidates_h, count_true_candidates_h))
        print("Done refinement!")
        print("Acc_e with max score_e: {:.4f} is : {}".format(refinement_model.score_max_e, acc_max_e))
        print("Acc_h with max score_h: {:.4f} is : {}".format(refinement_model.score_max_h, acc_max_h))
        refinement_model.alpha_source_e = alpha_source_max_e
        refinement_model.alpha_target_e = alpha_target_max_e
        refinement_model.alpha_source_h = alpha_source_max_h
        refinement_model.alpha_target_h = alpha_target_max_h

        self.CINA_S_e = S_max_e
        self.CINA_S_h = S_max_h
        self.log_and_evaluate(CINA, refinement_model, source_A_hat, target_A_hat)
        return self.CINA_S_e, self.CINA_S_h


    def train_CINA(self, CINA, source_A_hat, target_A_hat, structural_optimizer, threshold):
        CINA = self.gcn_semisup_training(CINA, source_A_hat, target_A_hat, structural_optimizer)
        print("Done structural training")
        CINA_S_e, CINA_S_h = self.refine(CINA, source_A_hat, target_A_hat, threshold)
        return CINA_S_e /4, CINA_S_h / 4

    def train_HCINA(self, HCINA, structural_optimizer, hg_s, hg_t):
        HCINA = self.hyper_gcn_semisup_training(HCINA, structural_optimizer, hg_s, hg_t)
        print("Done structural training")
        S_hyper = self.refine_hyper(HCINA, hg_s, hg_t)
        return S_hyper/4


    def get_similarity_matrices(self, source_outputs, target_outputs):
        """
        Construct Similarity matrix in each layer
        :params source_outputs: List of embedding at each layer of source graph
        :params target_outputs: List of embedding at each layer of target graph
        """
        list_S_e = []
        list_S_h = []
        for i in range(len(source_outputs)):
            source_output_i_e = source_outputs[i][0]
            source_output_i_h = source_outputs[i][1]
            target_output_i_e = target_outputs[i][0]
            target_output_i_h = target_outputs[i][1]
            S_e = torch.mm(F.normalize(source_output_i_e), F.normalize(target_output_i_e).t())
            S_h = torch.mm(F.normalize(source_output_i_h), F.normalize(target_output_i_h).t())
            list_S_e.append(S_e)
            list_S_h.append(S_h)
        return list_S_e[1:],list_S_h[1:]
        

    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'),self.args.w_mul_s_A, 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'),self.args.w_mul_t_A, 't')
        return source_outputs, target_outputs
    

    def get_candidate(self, source_outputs, target_outputs, threshold):
        List_S_e, List_S_h = self.get_similarity_matrices(source_outputs, target_outputs)
        source_candidates_e = []
        target_candidates_e = []
        source_candidates_h = []
        target_candidates_h = []
        count_true_candidates_e = 0
        count_true_candidates_h = 0
        if len(List_S_e) < 2:
            print("The current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates_e), torch.LongTensor(target_candidates_e)

        num_source_nodes = len(self.source_dataset.G.nodes())
        num_target_nodes = len(self.target_dataset.G.nodes())
        # e
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S_e)):
                if List_S_e[j][i].argmax() != List_S_e[j-1][i].argmax():
                    node_i_is_stable = False
                    break
            if node_i_is_stable:
                tg_candi = List_S_e[-1][i].argmax()

                source_candidates_e.append(i)
                target_candidates_e.append(tg_candi)
                try:
                    if self.full_dict[i] == tg_candi:
                        count_true_candidates_e += 1
                except:
                    continue
        # h
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S_h)):
                if List_S_h[j][i].argmax() != List_S_h[j - 1][i].argmax():
                    node_i_is_stable = False
                    break
            if node_i_is_stable:
                tg_candi = List_S_h[-1][i].argmax()

                source_candidates_h.append(i)
                target_candidates_h.append(tg_candi)
                try:
                    if self.full_dict[i] == tg_candi:
                        count_true_candidates_h += 1
                except:
                    continue

        return torch.LongTensor(source_candidates_e), torch.LongTensor(target_candidates_e), len(source_candidates_e), count_true_candidates_e, torch.LongTensor(source_candidates_h), torch.LongTensor(target_candidates_h), len(source_candidates_h), count_true_candidates_h

