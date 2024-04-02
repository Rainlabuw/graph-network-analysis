import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import control as ct 

eig = lambda A: np.linalg.eig(A)[0]
round = lambda A: np.round(A, 3)


def generate_random_connected_graph(n):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(0, n))

    print(G.nodes)

    # Connect the nodes randomly until the graph is connected
    while not nx.is_connected(G):
        # Choose two random nodes

        node1 = np.random.randint(0, n)
        node2 = np.random.randint(0, n)
        
        # Add an edge between the two nodes if it doesn't already exist
        if not G.has_edge(node1, node2) and node1 != node2:
            G.add_edge(node1, node2)

    return G

def example_graph():
    edges = [(0, 6), (0, 1), (1, 7), (1, 9), (2, 4), (2, 5), (3, 9), (4, 7), (5, 8), (5, 9), (6, 9), (6, 7), (7, 9)]
    G = nx.Graph(edges)
    return G, 10

def K4():
    """Complete graph with 4 nodes"""
    edges = [(0,1),(1,2),(2,3),(0,3),(0,2),(1,3)]
    G = nx.Graph(edges)
    return G, 4

def C4():
    """Cycle graph with 4 nodes"""
    edges = [(0,1),(1,2),(2,3),(0,3)]
    G = nx.Graph(edges)
    return G,4

def S4():
    """Star graph with 4 nodes"""
    edges = [(0,1),(0,2),(0,3)]
    G = nx.Graph(edges)
    return G,4

def P4():
    """Path graph with 4 nodes"""
    edges = [(0,1),(1,2),(2,3)]
    G = nx.Graph(edges)
    return G,4

G, n = example_graph()

L = nx.laplacian_matrix(G).toarray()
I = np.eye(n)

A = -L
plt.figure()
nx.draw(G, with_labels=True)

# randomly samples 2 pairs of nodes for input-output
while True:
    i1, i2, o1, o2 = np.random.randint(0,n,4)
    if i1 != o1 and i2 != o2:
        break

B1 = np.zeros(n)
B1[i1] = 1
B2 = np.zeros(n)
B2[i2] = 1

C1 = np.zeros(n)
C1[o1] = 1
C2 = np.zeros(n)
C2[o2] = 1

sys1 = ct.ss(A,B1,C1,0)
sys2 = ct.ss(A,B2,C2,0)

omega = np.logspace(-2, 2, 1000)
mag, phase, _ = ct.freqresp(sys1, omega)

mag1_dB = 20*np.log10(mag)
phase1 = phase/np.pi*180

mag, phase, _ = ct.freqresp(sys2, omega)
mag2_dB = 20*np.log10(mag)
phase2 = phase/np.pi*180

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(omega, mag1_dB, label=f"u={i1}, y={o1}")
plt.semilogx(omega, mag2_dB, label=f"u={i2}, y={o2}")
plt.grid()
plt.xlabel('freq (rad/s)')
plt.ylabel('mag (dB)')
plt.legend()

plt.subplot(2,1,2)
plt.semilogx(omega, phase1, label=f"u={i1}, y={o1}")
plt.semilogx(omega, phase2, label=f"u={i2}, y={o2}")
plt.grid()
plt.xlabel('freq (rad/s)')
plt.ylabel('phase (deg)')
plt.legend()


plt.show()
