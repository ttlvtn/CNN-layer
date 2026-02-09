import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps

st.set_page_config(page_title="ANN 內部機制解密", layout="wide")

st.title("🧠 深入 ANN 內部：每一層的物理運作與視覺化")
st.markdown("一個最基礎的人工神經網路 (ANN) 有三層。讓我們來看看每一層裡面到底發生了什麼魔法。")

# --- 1. 輸入層 (Input Layer) ---
st.header("1. 📍 輸入層：數據的『攤平』")
st.subheader("物理含意：將圖像轉為一維數字流")
st.write("這是 AI 大腦接收原始數據的第一步。對於圖片來說，ANN 會粗暴地把 2D 圖像『壓扁』成一長串 1D 數字。這時，圖片的空間關係就消失了。")

uploaded_file = st.file_uploader("請上傳一張圖片，觀看 ANN 的輸入層處理：", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    col_raw, col_flat = st.columns(2)
    
    with col_raw:
        st.image(img.resize((100, 100)), caption="原始圖片 (縮小)", width=100)
        
    with col_flat:
        img_gray = img.resize((50, 50)).convert('L') # 轉為灰階並縮小以利視覺化
        pixels = np.array(img_gray).flatten()
        
        fig_input, ax_input = plt.subplots(figsize=(6, 2))
        ax_input.plot(pixels[:200], color='skyblue', linewidth=0.8) # 顯示前200個像素點
        ax_input.set_title("被『攤平』的像素數字流")
        ax_input.set_xlabel("像素點編號")
        ax_input.set_ylabel("亮度值 (0-255)")
        st.pyplot(fig_input)
        st.caption("每個點都代表一個像素的亮度值，但它們已經沒有『左右鄰居』的關係了。")
        st.success("✅ **物理結論：** 輸入層是原始數據的搬運工，將 2D 資訊『去結構化』。")

st.markdown("---")

# --- 2. 隱藏層 (Hidden Layer) ---
st.header("2. 📍 隱藏層：模式的『過濾』與『激活』")
st.subheader("物理含意：神經元進行加權投票與開關決策")
st.write("這是 ANN 的『大腦核心』，每個神經元都像一個小小的決策者。")

col_weights, col_relu = st.columns(2)

with col_weights:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Artificial_neural_network.svg/1200px-Artificial_neural_network.svg.png", 
             caption="神經元連結與權重（示意圖）", width=300)
    st.write("**每個神經元都裝了：**")
    st.write("👉 **權重 (Weights)**：決定前一層哪個訊號最重要。")
    st.write("👉 **偏置 (Bias)**：調整神經元被激發的門檻。")
    st.success("✅ **物理動作：** 進行 $Weight \times Input + Bias$ 運算，篩選重要資訊。")

with col_relu:
    st.subheader("激活函數 (ReLU)：神經元的『開關』")
    st.write("這是決定神經元是否要把訊號傳遞出去的關鍵。")
    
    # 模擬 ReLU 視覺化
    x_relu = np.linspace(-3, 3, 100)
    y_relu = np.maximum(0, x_relu)
    
    fig_relu, ax_relu = plt.subplots(figsize=(5, 3))
    ax_relu.plot(x_relu, y_relu, color='purple', linewidth=2)
    ax_relu.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_relu.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_relu.set_title("ReLU 激活函數：負值歸零")
    ax_relu.set_xlabel("輸入值 (加權總和)")
    ax_relu.set_ylabel("輸出值")
    st.pyplot(fig_relu)
    st.caption("只有當輸入值大於零時，訊號才會被傳遞。否則，該訊號會被『關閉』。")
    st.success("✅ **物理結論：** 隱藏層是模式的過濾與激活中心，只讓重要的特徵訊號傳遞。")


st.markdown("---")

# --- 3. 輸出層 (Output Layer) ---
st.header("3. 📍 輸出層：最終的『決策投票』")
st.subheader("物理含意：將得分轉化為機率")
st.write("這是 ANN 做出最終判斷的地方。它會綜合所有隱藏層傳來的線索，然後告訴你機率最高的答案。")

# 模擬 Softmax 機率分佈
labels_output = ["類別 A", "類別 B", "類別 C"]
scores_output = np.array([st.slider(f"類別 {chr(65+i)} 的分數 (原始輸出)", -5.0, 5.0, float(i)) for i in range(3)])

exp_scores = np.exp(scores_output - np.max(scores_output)) # 避免溢出
probabilities = exp_scores / np.sum(exp_scores)

fig_output, ax_output = plt.subplots(figsize=(6, 3))
ax_output.bar(labels_output, probabilities, color=['#FFC107', '#2196F3', '#4CAF50'])
ax_output.set_ylim(0, 1)
ax_output.set_ylabel("機率 (%)")
ax_output.set_title("Softmax 輸出：機率分佈")
for i, prob in enumerate(probabilities):
    ax_output.text(i, prob + 0.05, f"{prob:.2%}", ha='center', color='black')
st.pyplot(fig_output)
st.caption("經過 Softmax 處理後，所有分數都會轉化為機率，總和為 100%。")
st.success("✅ **物理結論：** 輸出層是決策的終點，將內部得分轉換為外部可理解的機率預測。")

st.markdown("---")
st.header("💡 總結：ANN 的局限與 CNN 的進化")
st.write("""
- **ANN 的局限：** 由於『攤平』資料，ANN 失去了圖片的**空間感**，導致在處理圖像時效率不高。
- **CNN 的進化：** 透過引入**卷積層**，CNN 讓神經網路重新獲得了『視覺』。卷積層在 ANN 前面進行『特徵掃描』，讓隱藏層處理的不再是混亂的數字，而是有意義的『局部特徵』。
""")
st.write("這就是為什麼像 ResNet 這樣強大的影像 AI，都是 CNN 而不是純粹的 ANN。")
