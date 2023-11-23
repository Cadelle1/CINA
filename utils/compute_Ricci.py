from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

def compute_Ricci(dataset, edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G_OT = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
    G = G_OT.compute_ricci_curvature()

    ricci_list = []
    for n1, n2 in G.edges():
        ricci_list.append([n1, n2, G[n1][n2]['ricciCurvature']])
        ricci_list.append([n2, n1, G[n1][n2]['ricciCurvature']])

    # node_num = G.number_of_nodes()
    # for i in range(node_num):
    #     ricci_list.append([i, i, 0.0])

    ricci_list = sorted(ricci_list)
    Save_list(ricci_list, dataset)
    return  ricci_list


def Save_list(list, filename):
    file1 = open(filename + '.txt', 'w')
    for i in range(len(list)):
        for j in range(len(list[i])):
            file1.write(str(list[i][j]))
            file1.write(" ")
        file1.write('\n')
    file1.close()