import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps

# 頁面配置
st.set_page_config(page_title="AI 大腦解密實驗室", layout="wide")

st.title("🔬 AI 大腦全層級解密：從基礎 ANN 到 Transformer")
st.write("這是一個互動式實驗室，讓我們一層層拆解 AI 是如何『思考』的。")

# --- 側邊欄：導覽控制 ---
with st.sidebar:
    st.header("🛠️ 實驗工具箱")
    mode = st.radio("選擇教學階段：", 
                    ["1. ANN 基礎結構", "2. 隱藏層微觀運算", "3. CNN 進化視覺", "4. Transformer 全局視野"])
    st.markdown("---")
    st.info("💡 **核心物理含意**：\n\n**ANN**：數據大雜燴\n\n**CNN**：局部偵探\n\n**Transformer**：全局導演")

# --- 階段 1：ANN 基礎結構 ---
if mode == "1. ANN 基礎結構":
    st.header("📍 ANN：人工神經網路的骨架")
    
    tab1, tab2, tab3 = st.tabs(["輸入層 (Input)", "隱藏層 (Hidden)", "輸出層 (Output)"])
    
    with tab1:
        st.subheader("物理動作：數據攤平 (Flattening)")
        st.write("將 2D 圖片『壓扁』成 1D 線條。對 ANN 來說，圖片的空間感消失了。")
        test_img = np.random.randint(0, 255, (10, 10))
        col_a, col_b = st.columns(2)
        col_a.image(test_img.astype(np.uint8), caption="2D 像素矩陣", width=200)
        col_b.line_chart(test_img.flatten())
        
    with tab2:
        st.subheader("物理動作：特徵加權與過濾")
        st.write("隱藏層的神經元會相互連接，每個連結都有一個『權重』。")
        # 用繪圖取代外部圖片
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Input ➔ [ Weights ] ➔ Activation", ha='center', va='center', fontsize=12, bbox=dict(facecolor='orange', alpha=0.3))
        ax.axis('off')
        st.pyplot(fig)
        st.info("💡 權重代表資訊的重要性，由 Gradient（梯度）在訓練中不斷修正。")
        
    with tab3:
        st.subheader("物理動作：最後決策 (Softmax)")
        scores = np.array([8.0, 2.0, 1.0])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        st.bar_chart({"類別": ["貓", "狗", "鳥"], "機率": probs}, x="類別", y="機率")

# --- 階段 2：隱藏層微觀運算 ---
elif mode == "2. 隱藏層微觀運算":
    st.header("🔍 隱藏層的神經元開關")
    col_math, col_logic = st.columns(2)
    with col_math:
        input_v = st.slider("輸入訊號", -5.0, 5.0, 2.0)
        weight_v = st.slider("權重 (重要性)", -2.0, 2.0, 0.8)
        z = input_v * weight_v
        activated = max(0, z)
        st.latex(r"Output = ReLU(Weight \times Input)")
    with col_logic:
        fig, ax = plt.subplots(figsize=(4, 3))
        x_relu = np.linspace(-5, 5, 100)
        ax.plot(x_relu, np.maximum(0, x_relu), color='orange')
        ax.scatter([z], [activated], color='red')
        ax.set_title("ReLU 激活：負值歸零")
        st.pyplot(fig)
    
# --- 階段 3：CNN 進化視覺 ---
elif mode == "3. CNN 進化視覺":
    st.header("👁️ CNN：具有視覺結構的 AI")
    up = st.file_uploader("上傳圖片...", type=["jpg", "png"])
    if up:
        img = Image.open(up).convert('RGB')
        c1, c2 = st.columns(2)
        c1.image(img.convert('L').filter(ImageFilter.FIND_EDGES), caption="卷積層提取邊緣")
        c2.image(img.filter(ImageFilter.CONTOUR), caption="零件特徵提取")
        
# --- 階段 4：Transformer 全局視野 ---
else:
    st.header("⚡ Transformer：拼圖與全局關注")
    st.write("不掃描，直接把圖片切成拼圖塊。")
    up_t = st.file_uploader("上傳圖片...", type=["jpg", "png"], key="t")
    if up_t:
        img_t = np.array(Image.open(up_t).resize((224, 224)))
        for i in range(0, 224, 32):
            img_t[i:i+2, :, :] = 255
            img_t[:, i:i+2, :] = 255
        st.image(img_t, caption="Transformer 眼中的 Patches")
        
st.markdown("---")
st.caption("AI 教育實驗室 - 教學專用 (已修正環境錯誤)")
