from evaluation.metrics import get_statistics
import numpy as np
import torch
import pickle
from scipy.sparse import coo_matrix
import torch.nn.functional as F

def get_equivalent_edges(source_edges, target_edges, full_dict):
    count_edges = 0
    source_edges_list = []
    target_edges_list = []
    source_edges = source_edges.tolist()
    target_edges = target_edges.tolist()
    while count_edges < 100:
        index = np.random.randint(0, len(source_edges), 1)[0]
        source_edge = source_edges[index]
        if source_edge not in source_edges_list:
            first_node = source_edge[0]
            second_node = source_edge[1]
            try:
                first_node_target = full_dict[first_node]
                second_node_target = full_dict[second_node]
            except:
                continue
            if [first_node_target, second_node_target] in target_edges:
                source_edges_list.append(source_edge)
                target_edges_list.append([first_node_target, second_node_target])
                count_edges += 1
    
    source_nodes = np.random.choice(np.array(list(full_dict.keys())), 100, replace=False)
    target_nodes = np.array([full_dict[source_nodes[i]] for i in range(len(source_nodes))])

    return source_edges_list, target_edges_list, source_nodes, target_nodes

def investigate(source_outputs, target_outputs, source_edges, target_edges, full_dict):
    source_edges, target_edges, source_nodes, target_nodes = get_equivalent_edges(source_edges, target_edges, full_dict)
    source_edges_np = np.array(source_edges)
    target_edges_np = np.array(target_edges)

    source_nodes_np = np.array(source_nodes)
    target_nodes_np = np.array(target_nodes)
    first_source_nodes_np = source_edges_np[:, 0]
    second_source_nodes_np = source_edges_np[:, 1]
    first_target_nodes_np = target_edges_np[:, 0]
    second_target_nodes_np = target_edges_np[:, 1]

    source_nodes_tensor = torch.LongTensor(source_nodes_np).cuda()
    target_nodes_tensor = torch.LongTensor(target_nodes_np).cuda()
    first_source_nodes_tensor = torch.LongTensor(first_source_nodes_np).cuda()
    second_source_nodes_tensor = torch.LongTensor(second_source_nodes_np).cuda()
    first_target_nodes_tensor = torch.LongTensor(first_target_nodes_np).cuda()
    second_target_nodes_tensor = torch.LongTensor(second_target_nodes_np).cuda() 

    source_nodes_emb = [source_outputs[i][source_nodes_tensor] for i in range(len(source_outputs))]
    target_nodes_emb = [target_outputs[i][target_nodes_tensor] for i in range(len(source_outputs))]
    first_source_nodes_emb = [source_outputs[i][first_source_nodes_tensor] for i in range(len(source_outputs))]
    second_source_nodes_emb = [source_outputs[i][second_source_nodes_tensor] for i in range(len(source_outputs))]
    first_target_nodes_emb = [target_outputs[i][first_target_nodes_tensor] for i in range(len(source_outputs))]
    second_target_nodes_emb = [target_outputs[i][second_target_nodes_tensor] for i in range(len(source_outputs))]

    edges_distance_source = [torch.sum((first_source_nodes_emb[i] - second_source_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    edges_distance_target = [torch.sum((first_target_nodes_emb[i] - second_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    anchor_distance1 = [torch.sum((first_source_nodes_emb[i] - first_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    anchor_distance2 = [torch.sum((second_source_nodes_emb[i] - second_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    random_distance1 = [torch.sum((first_source_nodes_emb[i] - source_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    random_distance2 = [torch.sum((first_target_nodes_emb[i] - target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]

    for i in range(len(edges_distance_source)):
        print("Layer: {}, edge source: {:.4f}, edge target: {:.4f}, non edge source: {:.4f}, non edge target: {:.4f}".format(i, edges_distance_source[i].mean(), edges_distance_target[i].mean(), \
                random_distance1[i].mean(), random_distance2[i].mean()))
        print("Layer: {}, anchor distance1: {:.4f}, anchor distance2: {:.4f}".format(i, anchor_distance1[i].mean(), anchor_distance2[i].mean()))


def get_acc(source_outputs, target_outputs, test_dict = None, alphas_e=None, alphas_h=None):

    Sf_e = np.zeros((len(source_outputs[0][0]), len(target_outputs[0][0])))
    Sf_h = np.zeros((len(source_outputs[0][1]), len(target_outputs[0][1])))
    accs_e = ""
    accs_h = ""
    for i in range(0, len(source_outputs)):
        S_e = torch.matmul(F.normalize(source_outputs[i][0]), F.normalize(target_outputs[i][0]).t())
        S_e_numpy = S_e.detach().cpu().numpy()
        if test_dict is not None:
            acc_e = get_statistics(S_e_numpy, test_dict)
            accs_e += "Acc_e layer {} is: {:.4f}, ".format(i, acc_e)
        if alphas_e is not None:
            Sf_e += alphas_e[i] * S_e_numpy
        else:
            Sf_e += S_e_numpy

        S_h = torch.matmul(F.normalize(source_outputs[i][1]), F.normalize(target_outputs[i][1]).t())
        S_h_numpy = S_h.detach().cpu().numpy()
        if test_dict is not None:
            acc_h = get_statistics(S_h_numpy, test_dict)
            accs_h += "Acc_h layer {} is: {:.4f}, ".format(i, acc_h)
        if alphas_h is not None:
            Sf_h += alphas_h[i] * S_h_numpy
        else:
            Sf_h += S_h_numpy

    if test_dict is not None:
        acc_e = get_statistics(Sf_e, test_dict)
        acc_h = get_statistics(Sf_h, test_dict)
        accs_e += "Final acc_e is: {:.4f}".format(acc_e)
        accs_h += "Final acc_h is: {:.4f}".format(acc_h)
    return accs_e, accs_h, Sf_e, Sf_h

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


def Laplacian_graph(A):
    for i in range(len(A)):
        A[i, i] = 1
    A = torch.FloatTensor(A)
    D_ = torch.diag(torch.sum(A, 0)**(-0.5))
    A_hat = torch.matmul(torch.matmul(D_,A),D_)
    A_hat = A_hat.float()
    indices = torch.nonzero(A_hat).t()
    values = A_hat[indices[0], indices[1]]
    A_hat = torch.sparse.FloatTensor(indices, values, A_hat.size())
    return A_hat, coo_matrix(A.detach().cpu().numpy())

def update_Laplacian_graph(old_A, new_edges):
    count_updated = 0
    for edge in new_edges:
        if old_A[edge[0], edge[1]] == 0:
            count_updated += 1
        old_A[edge[0], edge[1]] = 1
        old_A[edge[1], edge[0]] = 1
    new_A_hat, new_A = Laplacian_graph(old_A)
    print("Updated {} edges".format(count_updated))
    return new_A_hat, new_A


def get_candidate_edges(S, edges, threshold):
    S = S / 3
    points_source, points_source_index = S[edges[:, 0]].max(dim=1)
    points_target, points_target_index = S[edges[:, 1]].max(dim=1)
    new_edges = []
    for i in range(len(points_source)):
        point_source = points_source[i]
        point_target = points_target[i]
        if point_source > threshold and point_target > threshold:
            new_edges.append((points_source_index[i], points_target_index[i]))
    return new_edges


def get_source_target_neg(args, source_deg, target_deg, source_edges, target_edges):
    source_negs = []
    target_negs = []
    for i in range(0, len(source_edges), 512):
        source_neg = fixed_unigram_candidate_sampler(
                num_sampled=args.neg_sample_size,
                unique=False,
                range_max=len(source_deg),
                distortion=0.75,
                unigrams=source_deg
                )
        
        source_neg = torch.LongTensor(source_neg).cuda()
        source_negs.append(source_neg)

    for i in range(0 ,len(target_edges), 512):

        target_neg = fixed_unigram_candidate_sampler(
            num_sampled=args.neg_sample_size,
            unique=False,
            range_max=len(target_deg),
            distortion=0.75,
            unigrams=target_deg
            )

        target_neg = torch.LongTensor(target_neg).cuda()
        target_negs.append(target_neg)

    return source_negs, target_negs       


def save_embeddings(source_outputs, target_outputs):
    print("Saving embeddings")
    for i in range(len(source_outputs)):
        ele_source = source_outputs[i]
        ele_source = ele_source.detach().cpu().numpy()
        ele_target = target_outputs[i]
        ele_target = ele_target.detach().cpu().numpy()
        np.save("numpy_emb/source_layer{}".format(i), ele_source)
        np.save("numpy_emb/target_layer{}".format(i), ele_target)
    print("Done saving embeddings")


def investigate_similarity_matrix(S, full_dict, source_deg, target_deg, source_edges, target_edges):
    source_nodes = np.array(list(full_dict.keys()))
    target_nodes = np.array(list(full_dict.values()))
    hits_source = []
    hits_target = []
    for i in range(len(S)):
        S_i = S[i][source_nodes]
        matched_source = np.argmax(S_i, axis=1)
        hit_i_source = []
        hit_i_target = []
        for j in range(len(source_nodes)):
            if matched_source[j] == target_nodes[j]:
                hit_i_source.append(source_nodes[j])
                hit_i_target.append(target_nodes[j])
        hits_source.append(hit_i_source)
        hits_target.append(hit_i_target)
    
    tosave = [hits_source, hits_target]
    with open("douban_data", "wb") as f:
        pickle.dump(tosave, f)

    
    for i in range(len(hits_source)):
        source_deg_i = np.array([source_deg[k] for k in hits_source[i]])
        target_deg_i = np.array([target_deg[k] for k in hits_target[i]])
        mean_source_i, mean_target_i, std_source_i, std_target_i = degree_distribution(source_deg_i, target_deg_i)
        print("Layer: {} MEAN source: {}, target: {}. STD source: {}, target: {}".format(i + 1, mean_source_i, mean_target_i, std_source_i, std_target_i))



def degree_distribution(source_deg, target_deg):
    if False:
        for i in range(len(source_deg)):
            print("Source degree: {}, target degree: {}".format(source_deg[i], target_deg[i]))
    
    
    mean_source_deg = np.mean(source_deg)
    mean_target_deg = np.mean(target_deg)
    std_source_deg = np.std(source_deg)
    std_target_deg = np.std(target_deg)

    return mean_source_deg, mean_target_deg, std_source_deg, std_target_deg
    

def get_nn_avg_dist(simi, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """

    best_simi_indice = np.argpartition(simi, -knn)[:, -knn:]
    best_simi_value = np.array([simi[i, best_simi_indice[i]] for i in range(len(best_simi_indice))]).mean(axis=1).reshape(len(best_simi_indice), 1)

    return best_simi_value

def get_candidates(simi):
    """
    Get best translation pairs candidates.
    """
    knn = '10'
    assert knn.isdigit()
    knn = int(knn)
    average_dist1 = get_nn_avg_dist(simi, knn)
    average_dist2 = get_nn_avg_dist(simi.T, knn)
    score = 2 * simi
    score = score - average_dist1 - average_dist2
    return score