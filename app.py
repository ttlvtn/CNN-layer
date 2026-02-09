import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import requests
import torch
from torchvision import models, transforms

# 頁面設定
st.set_page_config(page_title="AI 大腦解密：從 ANN 到 CNN", layout="wide")

# --- 1. 模型與標籤載入 (用於 CNN 演示) ---
@st.cache_resource
def get_cnn_resources():
    res101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    res101.eval()
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.split("\n")
    return res101, labels

resnet_model, imagenet_labels = get_cnn_resources()

# --- 2. 介面標題 ---
st.title("🧠 AI 大腦解密：從『數字感』到『空間感』")
st.markdown("""
我們將探討兩種 AI 大腦：
1.  **ANN (人工神經網路)**：最基礎的大腦，擅長處理數字表格。
2.  **CNN (卷積神經網路)**：**ANN 的進化版**，增加了『視覺』能力，專為圖像設計。
""")

# --- 3. 圖片上傳區 ---
uploaded_file = st.file_uploader("📸 上傳一張圖片，探索 AI 的思考過程...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    st.image(img, caption="原始輸入圖片", use_container_width=True)
    st.markdown("---")

    col_ann_viz, col_cnn_viz = st.columns(2)

    # --- ANN 的運作邏輯與視覺化 ---
    with col_ann_viz:
        st.header("1. ANN：『數字大雜燴』")
        st.subheader("物理含意：**資訊攤平與加權投票**")
        st.write("ANN 就像一群只會看數字的會計師。它會把圖片的像素**完全攤平**成一長串數字，然後每個數字都去影響下一層的每個神經元。")
        st.write("這會讓 AI 喪失圖片的**空間感**（上下左右關係）。")

        # 模擬 ANN 的攤平與全連接層
        img_arr_gray = np.array(img.resize((50, 50)).convert('L')) # 縮小並轉灰度
        flattened_data = img_arr_gray.flatten() # 攤平為一維
        
        fig_ann, ax_ann = plt.subplots(figsize=(6, 3))
        ax_ann.plot(flattened_data[:200], color='skyblue') # 只顯示前 200 個點
        ax_ann.set_title("ANN 眼中的圖片 (一維數字流)")
        ax_ann.set_xlabel("像素點序號")
        ax_ann.set_ylabel("像素亮度值")
        st.pyplot(fig_ann)
        st.caption("AI 看到的是一串沒有空間意義的數字，然後透過『權重』進行複雜的加權投票。")

    # --- CNN 的運作邏輯與視覺化 ---
    with col_cnn_viz:
        st.header("2. CNN：『視覺偵探』")
        st.subheader("物理含意：**濾鏡掃描與局部特徵**")
        st.write("CNN 是 **ANN 的進化版**。它在最前面加上了**卷積層 (Convolutional Layer)**，就像給 AI 裝上了許多『偵探濾鏡』。")
        st.write("每個濾鏡專門負責在圖片上掃描，尋找特定的局部特徵（例如：邊緣、紋理、小形狀）。")

        # 模擬 CNN 的卷積層處理 (邊緣檢測)
        cnn_edge_view = img.convert('L').filter(ImageFilter.FIND_EDGES)
        st.image(cnn_edge_view, caption="CNN 前半段：卷積層提取『邊緣』特徵", use_column_width=True)
        st.caption("卷積層能夠保留圖片的『空間感』，它知道線條在哪裡。")
        
        st.markdown("---")
        st.subheader("✨ CNN 的後半段：還是 ANN！")
        st.write("在卷積層提取完所有特徵後，CNN 會把這些**特徵圖**『攤平』，然後送入傳統 ANN 的**全連接層 (Fully Connected Layer)** 進行最終的判斷和分類。")
        st.write("可以理解為：**CNN = 『卷積偵探組』 + 『ANN 投票部隊』**")

    st.markdown("---")

    # --- CNN 最終辨識結果 (作為 CNN 實際應用的例子) ---
    st.header("🏆 CNN 辨識結果 (ResNet-101 示範)")
    st.info("ResNet-101 是一個超級強大的 CNN，它有超過 100 層卷積層來提取特徵，最後再用 ANN 判斷。")
    
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = resnet_model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_id = torch.topk(prob, 1)

    st.success(f"結果：**{imagenet_labels[top_id[0]]}** (信心指數：{top_prob[0]:.2%})")
    
    st.markdown("---")
    st.subheader("總結：AI 的進化之路")
    st.write("""
    - **ANN** 學習數字模式，但對圖像空間不敏感。
    - **CNN** 透過卷積層獲得『視覺』，能有效處理圖像，成為今天最主流的影像 AI。
    - 像 ResNet-101 或 EfficientNet 這些強大的模型，都是 CNN 的代表，它們在『卷積層』的設計上各有巧妙！
    """)
    
    st.balloons()
