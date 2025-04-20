#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 22:20:48 2025

@author: supatsarasaennang
"""


# kmeans_app_with_model.py

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ตั้งชื่อแอป
st.title("🔍 K-Means Clustering App with Trained Models")

# โหลดโมเดลที่ฝึกไว้แล้ว
with open("dtm_trained_model.pkl", "rb") as f:
    X = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# ทำนาย label จากโมเดล
labels = kmeans.predict(X)

# ลดมิติด้วย PCA เพื่อแสดงผล
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
for cluster in sorted(df['Cluster'].unique()):
    clustered = df[df['Cluster'] == cluster]
    ax.scatter(clustered['PCA1'], clustered['PCA2'], label=f'Cluster {cluster}')
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Clusters (2D PCA Projection)")
ax.legend()

# แสดงกราฟใน Streamlit
st.pyplot(fig)

