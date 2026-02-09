import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI 視覺進化：從 ANN 到 CNN", layout="wide")

st.title("🧩 AI 大腦拆解：ANN 與 CNN 的物理意義")

uploaded_file = st.file_uploader("上傳一張圖片，觀測隱藏層的物理變化...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    col_ann, col_cnn = st.columns(2)

    # --- ANN 視覺化：呈現「數據大雜燴」 ---
    with col_ann:
        st.header("1. ANN 模式 (隱藏層基礎)")
        st.write("🔴 **物理含意：數據攤平 (Flattening)**")
        st.info("ANN 把圖片壓扁成一維數字。它失去空間感，只靠『權重投票』來找規律。")
        
        # 模擬圖片攤平
        img_gray = img.resize((50, 50)).convert('L')
        pixels = np.array(img_gray).flatten()
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(pixels[:500], color='gray', linewidth=0.5)
        ax.set_title("ANN 眼中的數字流 (Hidden Layer 輸入前)")
        st.pyplot(fig)
        st.caption("這就是隱藏層在處理的東西：一長串毫無章法的數字。")

    # --- CNN 視覺化：呈現「卷積濾鏡」 ---
    with col_cnn:
        st.header("2. CNN 模式 (套用卷積層)")
        st.write("🟢 **物理含意：特徵掃描 (Filtering)**")
        st.info("CNN 像戴上掃描眼鏡。它保留了空間結構，能認出『線條』與『形狀』。")
        
        # 模擬卷積提取邊緣
        cnn_view = img.convert('L').filter(ImageFilter.FIND_EDGES)
        st.image(cnn_view, caption="CNN 卷積層提取出的物理特徵 (邊緣圖)", use_container_width=True)
        st.caption("這就是 CNN 的優勢：它能看見形狀，而不只是數字。")

    st.markdown("---")

    # --- CNN 的三階段物理質變 ---
    st.header("🏗️ CNN 卷積層的層級進化")
    v1, v2, v3 = st.columns(3)

    with v1:
        st.subheader("第一階段：找線條")
        st.image(img.convert('L').filter(ImageFilter.FIND_EDGES), use_container_width=True)
        st.write("**物理含意**：偵測像素變化，找邊緣。")

    with v2:
        st.subheader("第二階段：找零件")
        part_view = img.filter(ImageFilter.SHARPEN).convert('RGB')
        st.image(part_view, use_container_width=True)
        st.write("**物理含意**：組合線條，變成耳朵、眼睛或輪胎。")

    with v3:
        st.subheader("第三階段：看邏輯")
        # 模擬熱力圖 (Attention Map)
        heatmap = img.convert('L').resize((14, 14)).resize(img.size, resample=Image.NEAREST)
        heatmap = ImageOps.colorize(heatmap, black="blue", white="red")
        st.image(heatmap, use_container_width=True)
        st.write("**物理含意**：理解物件位置，決定這到底是什麼。")

st.write("---")
st.write("💡 **教學結語**：ANN 是靈魂，它透過隱藏層學會決策；卷積層是眼睛，它讓 ANN 學會看圖。兩者結合，就是我們今天看到的強大影像 AI！")
