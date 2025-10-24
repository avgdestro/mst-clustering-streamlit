import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Step 1: Upload data
st.title("MST-Based Graph Clustering App")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df)
    data = df.values[:, :2]  # use first two columns for simplicity

    # Step 2: Build distance matrix and MST
    dist_mat = distance_matrix(data, data)
    G = nx.Graph()
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            G.add_edge(i, j, weight=dist_mat[i][j])
    mst = nx.minimum_spanning_tree(G)

    # Step 3: Remove longest edges to create clusters
    st.write("Choose the number of clusters:")
    k = st.slider("Clusters", 2, len(data), 2)
    edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    edges_to_remove = edges[:k-1]  # remove k-1 longest edges

    mst_copy = mst.copy()
    for e in edges_to_remove:
        mst_copy.remove_edge(e[0], e[1])

    # Step 4: Show cluster assignments
    clusters = list(nx.connected_components(mst_copy))
    labels = np.zeros(len(data), dtype=int)
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            labels[node] = idx

    st.write("Cluster assignments (by index):", labels)

    # Step 5: Plot results
    fig, ax = plt.subplots()
    for idx, cluster in enumerate(clusters):
        points = data[list(cluster)]
        ax.scatter(points[:, 0], points[:, 1], label=f"Cluster {idx+1}")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)
