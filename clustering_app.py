#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:20:48 2025

@author: supatsarasaennang
"""


import streamlit as st
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# ตั้งชื่อแอป
st.title("🔍 K-Means Clustering App with Iris Dataset")

# โหลดข้อมูล Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# แถบด้านข้างสำหรับเลือกจำนวน cluster
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)

# สร้างโมเดล KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# ลดมิติด้วย PCA เพื่อแสดงผลแบบ 2D
pca = PCA(n_components=2)
components = pca.fit_transform(X)

# สร้าง DataFrame สำหรับพล็อตกราฟ
df = pd.DataFrame({
    'PCA1': components[:, 0],
    'PCA2': components[:, 1],
    'Cluster': labels
})

# พล็อตผลลัพธ์
fig, ax = plt.subplots()
for cluster in range(k):
    clustered = df[df['Cluster'] == cluster]
    ax.scatter(clustered['PCA1'], clustered['PCA2'], label=f'Cluster {cluster}')
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Clusters (2D PCA Projection)")
ax.legend()

# แสดงกราฟใน Streamlit
st.pyplot(fig)

