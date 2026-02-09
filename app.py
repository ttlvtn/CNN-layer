import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps

# 1. 頁面配置
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
    st.write("物理含意：將所有資訊轉化為數字與機率的過程。")
    
    tab1, tab2, tab3 = st.tabs(["輸入層 (Input Layer)", "隱藏層 (Hidden Layer)", "輸出層 (Output Layer)"])
    
    with tab1:
        st.subheader("物理動作：數據攤平 (Flattening)")
        st.write("將 2D 圖片『壓扁』成 1D 線條。對 ANN 來說，圖片的空間感消失了，變成了純粹的數字流。")
        test_img = np.random.randint(0, 255, (10, 10))
        col_a, col_b = st.columns(2)
        col_a.image(test_img.astype(np.uint8), caption="2D 像素矩陣", width=200)
        col_b.line_chart(test_img.flatten())
        st.caption("右圖是 ANN 真正看到的『數字流』。")
        

    with tab2:
        st.subheader("物理動作：特徵加權與過濾")
        st.write("隱藏層負責從數字流中找出規律。它會給重要的數字高分（權重），不重要的低分。")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Artificial_neural_network.svg/1200px-Artificial_neural_network.svg.png", width=500)
        

    with tab3:
        st.subheader("物理動作：最後決策 (Softmax)")
        st.write("將隱藏層的得分轉化為機率，機率總和必為 100%。")
        scores = np.array([10.0, 5.0, 2.0])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        st.bar_chart({"類別": ["貓", "狗", "汽車"], "機率": probs}, x="類別", y="機率")

# --- 階段 2：隱藏層微觀運算 ---
elif mode == "2. 隱藏層微觀運算":
    st.header("🔍 隱藏層裡面到底在幹嘛？")
    st.write("每個神經元都是一個『過濾開關』。")
    
    col_math, col_logic = st.columns(2)
    
    with col_math:
        st.subheader("運算公式：權重 + 激活")
        st.latex(r"Output = ReLU(Weight \times Input + Bias)")
        input_v = st.slider("輸入訊號強度 (來自前一層)", -5.0, 5.0, 2.0)
        weight_v = st.slider("當前權重 (由 Gradient 修正而來)", -2.0, 2.0, 0.8)
        bias_v = st.slider("偏置 (門檻值)", -2.0, 2.0, -0.5)
        
        z = input_v * weight_v + bias_v
        activated = max(0, z)
        
        
    with col_logic:
        st.subheader("物理狀態視覺化 (ReLU)")
        fig, ax = plt.subplots()
        x_relu = np.linspace(-5, 5, 100)
        y_relu = np.maximum(0, x_relu)
        ax.plot(x_relu, y_relu, color='orange', lw=2)
        ax.scatter([z], [activated], color='red', s=100, zorder=5)
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        ax.set_title("ReLU 激活：負值歸零 (過濾雜訊)")
        st.pyplot(fig)
        
        if activated > 0:
            st.success(f"🔥 訊號通過！強度 {activated:.2f}。這是重要特徵，往下一層送。")
        else:
            st.error("❄️ 訊號被攔截。強度為 0。這被判定為雜訊，不往後傳。")
    
    st.markdown("---")
    st.subheader("⚙️ 隱藏層的進階組件")
    st_a, st_b, st_c = st.columns(3)
    st_a.info("**Batch Norm**\n\n對齊數據，防止訊號強度亂跑。")
    st_b.info("**Dropout**\n\n隨機讓神經元罷工，防止 AI 死背答案。")
    st_c.info("**Skip Connection**\n\n快速捷徑，ResNet 101 層不迷路的靈魂。")
    

# --- 階段 3：CNN 進化視覺 ---
elif mode == "3. CNN 進化視覺":
    st.header("👁️ CNN：裝上『卷積層』的 AI 偵探")
    st.write("當神經網路套用卷積層，AI 就能看見空間結構，而不再是亂碼數字。")
    
    up = st.file_uploader("上傳一張圖片，拆解卷積層的物理處理...", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up).convert('RGB')
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("第一步：找邊緣")
            st.image(img.convert('L').filter(ImageFilter.FIND_EDGES), caption="卷積淺層：素描線條")
        with c2:
            st.subheader("第二步：找零件")
            st.image(img.filter(ImageFilter.SHARPEN), caption="卷積中層：局部特徵")
        with c3:
            st.subheader("第三步：看語意")
            heatmap = img.convert('L').resize((14,14)).resize(img.size, resample=Image.NEAREST)
            st.image(ImageOps.colorize(heatmap, "blue", "red"), caption="卷積深層：理解位置邏輯")
        

# --- 階段 4：Transformer 全局視野 ---
else:
    st.header("⚡ Transformer：全局關注機制")
    st.write("Transformer 取代 CNN 的關鍵：它不再滑動掃描，而是直接看整張圖的關聯。")
    
    st.subheader("物理含意：拼圖化 (Patching) + 自注意力 (Attention)")
    up_t = st.file_uploader("上傳圖片看看 Transformer 如何『切拼圖』...", type=["jpg", "png", "jpeg"])
    if up_t:
        # 將圖片切片視覺化
        img_t = np.array(Image.open(up_t).resize((224, 224)))
        p_size = 32
        for i in range(0, 224, p_size):
            img_t[i:i+2, :, :] = 255
            img_t[:, i:i+2, :] = 255
        st.image(img_t, caption="Transformer 眼中的拼圖序列 (Patches)")
        st.info("💡 **自注意力 (Self-Attention)**：每個拼圖塊會與其他所有拼圖同時比較，這讓 AI 瞬間看清全局關係。")
        

st.markdown("---")
st.caption("AI 教育實驗室 - 教學專用 (無氣球特效)")
