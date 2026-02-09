import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANN 內部構造探險", layout="wide")

st.title("🧠 ANN 每一層到底裝了什麼？")

# 模擬一個簡單的神經元運算
st.header("1. 隱藏層的神經元運算模擬")
st.write("每個神經元都在進行：$Output = Activation(Weight \times Input + Bias)$")

col1, col2 = st.columns(2)

with col1:
    input_val = st.slider("輸入訊號強度 (Input)", 0.0, 1.0, 0.5)
    weight_val = st.slider("權重設定 (Weight/重要性)", -2.0, 2.0, 1.2)
    bias_val = st.slider("偏置設定 (Bias/門檻)", -1.0, 1.0, -0.2)

with col2:
    # 簡單模擬 ReLU 激活函數
    z = input_val * weight_val + bias_val
    output_val = max(0, z)
    
    st.metric("神經元輸出強度", f"{output_val:.2f}")
    if output_val > 0:
        st.success("✅ 訊號成功激發！傳遞到下一層。")
    else:
        st.error("❌ 訊號太弱，被攔截了。")

st.markdown("---")

# 輸出層的邏輯
st.header("2. 輸出層：最終機率投票")
st.write("輸出層會把所有神經元的得分轉化為機率。")

labels = ["貓 (Cat)", "狗 (Dog)", "汽車 (Car)"]
scores = st.multiselect("手動設定輸出層得分：", [1, 2, 5, 8, 10], default=[8, 2, 1])

if len(scores) == 3:
    # 模擬 Softmax
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)
    
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color=['#ff9999','#66b3ff','#99ff99'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("機率 (%)")
    st.pyplot(fig)
else:
    st.warning("請選擇剛好 3 個得分數值。")

st.write("---")
st.info("💡 **教學點：** ANN 的層級是為了處理資料模式，而 CNN 加入的卷積層則是為了讓這些層級能更聰明地『看見』圖片特徵。")
