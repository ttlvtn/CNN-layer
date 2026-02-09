import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

st.set_page_config(page_title="AI 構造大解密", layout="wide")

st.title("🔬 AI 大腦層級拆解：從像素到決策")

# 1. 第一層分頁：ANN 與 CNN 的巨觀對比
tab_ann, tab_cnn = st.tabs(["基礎大腦：ANN (人工神經網路)", "進階視覺：CNN (卷積神經網路)"])

# --- ANN 分頁 ---
with tab_ann:
    st.header("🏢 ANN 結構：資訊處理工廠")
    st.write("ANN 處理資訊就像在做『數字大雜燴』的統計。")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📍 輸入層 (Input Layer)"):
            st.write("**物理含意**：將影像『攤平』。")
            st.write("**細部邏輯**：將像素 2D 矩陣轉為 1D 向量。")
            st.code("flattened = image.reshape(-1)")
            
    with col2:
        with st.expander("📍 隱藏層 (Hidden Layer)"):
            st.write("**物理含意**：特徵加權與過濾。")
            st.write("**細部邏輯**：$y = f(Wx + b)$")
            st.write("神經元透過權重(W)找模式，再由激活函數(f)過濾雜訊。")
            
    with col3:
        with st.expander("📍 輸出層 (Output Layer)"):
            st.write("**物理含意**：機率決策。")
            st.write("**細部邏輯**：使用 Softmax 將得分轉為 0~1 的機率。")
            st.write("例如：貓 (0.9), 狗 (0.1)")

# --- CNN 分頁 ---
with tab_cnn:
    st.header("👁️ CNN 進化：裝上濾鏡的眼睛")
    st.write("神經網路套用了**卷積層**後，就能看見『形狀』。")
    
    c_col1, c_col2, c_col3 = st.columns(3)
    
    with c_col1:
        with st.expander("🔍 卷積層 (Convolution)"):
            st.write("**物理含意**：局部掃描濾鏡。")
            st.write("**細部邏輯**：濾鏡(Kernel)在圖片上滑動做內積運算。")
            st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif", caption="濾鏡滑動模擬")
            
    with c_col2:
        with st.expander("📏 池化層 (Pooling)"):
            st.write("**物理含意**：重點摘要。")
            st.write("**細部邏輯**：縮小圖片尺寸，只保留區域內最強的訊號。")
            
    with c_col3:
        with st.expander("🧩 全連接層 (ANN 部份)"):
            st.write("**物理含意**：零件組合與最後投票。")
            st.write("**細部邏輯**：將特徵圖轉回 ANN 結構，根據零件特徵做最後決定。")

# --- 互動演示區 ---
st.markdown("---")
st.header("🎮 實戰演示：上傳圖片看濾鏡效果")
up_file = st.file_uploader("上傳圖片...", type=["jpg","png"])

if up_file:
    img = Image.open(up_file).convert('RGB')
    
    # 模擬 CNN 第一層 (找邊緣)
    st.subheader("CNN 第一層：偵探濾鏡正在尋找邊緣線條...")
    edge_img = img.convert('L').filter(ImageFilter.FIND_EDGES)
    st.image(edge_img, width=400)
    st.info("物理含意：這就是卷積層在隱藏層裡幹的好事！它把顏色去掉了，只留下物體的邊界資訊。")

st.info("💡 **教學點**：ResNet-101 其實就是重複了這個過程 101 次，讓 AI 能從微小的線條一直理解到複雜的物件語意。")
