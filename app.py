import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Step 1: App Title
st.title("MST-Based Graph Clustering App")

# Step 2: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # Ensure data has at least two numeric columns
    if df.shape[1] < 2:
        st.error("Please upload a CSV file with at least two numeric columns.")
    else:
        try:
            # Use only the first two numeric columns for clustering
            data = df.select_dtypes(include=[np.number]).iloc[:, :2].values

            # Step 3: Build distance matrix and MST
            dist_mat = distance_matrix(data, data)
            G = nx.Graph()
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    G.add_edge(i, j, weight=dist_mat[i][j])
            mst = nx.minimum_spanning_tree(G)

            # Step 4: Choose number of clusters
            st.write("### Choose the number of clusters:")
            k = st.slider("Clusters (k)", 2, min(len(data), 10), 2)

            # Remove the (k-1) longest edges
            edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            edges_to_remove = edges[:k - 1]

            mst_copy = mst.copy()
            for e in edges_to_remove:
                mst_copy.remove_edge(e[0], e[1])

            # Step 5: Assign cluster labels
            clusters = list(nx.connected_components(mst_copy))
            labels = np.zeros(len(data), dtype=int)
            for idx, cluster in enumerate(clusters):
                f
