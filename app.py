import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import requests

st.set_page_config(page_title="AI 大腦實驗室", layout="wide")

# --- 1. 載入模型與標籤 ---
@st.cache_resource
def load_all():
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).eval()
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.split("\n")
    return model, labels

model, labels = load_all()

# --- 2. 教學主題 ---
st.title("🧩 AI 視覺進化：從 ANN 到 CNN")
st.markdown("""
神經網路（ANN）是基礎，但它看圖片時像在看一堆亂碼。
當我們給它裝上**卷積層**，它就變成了具備『視覺特徵提取』能力的 **CNN**。
""")

uploaded_file = st.file_uploader("📸 上傳一張照片，拆解 AI 的思考邏輯...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    # 視覺化對比區
    col_ann, col_cnn = st.columns(2)

    # --- ANN 視覺化：呈現「大雜燴」邏輯 ---
    with col_ann:
        st.header("1. ANN 運作：數據攤平")
        st.write("**物理含意：資訊大雜燴**")
        st.write("ANN 會把 2D 圖片壓成 1D 線條。它看到的是數字的跳動，而非圖像。")
        
        # 將圖片轉為灰階並攤平
        img_gray = img.resize((50, 50)).convert('L')
        pixels = np.array(img_gray).flatten()
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(pixels[:500], color='gray', linewidth=0.5)
        ax.set_title("ANN 眼中的數字流")
        st.pyplot(fig)
        st.caption("這就是 ANN 進行『全連接投票』前的樣子。")

    # --- CNN 視覺化：呈現「卷積層」邏輯 ---
    with col_cnn:
        st.header("2. CNN 運作：特徵提取")
        st.write("**物理含意：局部濾鏡掃描**")
        st.write("CNN 保留了空間關係。濾鏡會找出邊緣和形狀，就像偵探在找線索。")
        
        # 修正後的 Filter 語法
        cnn_edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
        st.image(cnn_edges, caption="卷積層抓取的邊緣特徵圖", use_container_width=True)

    st.markdown("---")

    # --- 3. 語意質變呈現 ---
    st.header("🏗️ 卷積層的三階段質變")
    v1, v2, v3 = st.columns(3)
    
    with v1:
        st.subheader("第一階段：邊緣")
        st.image(img.convert('L').filter(ImageFilter.FIND_EDGES), use_container_width=True)
        st.write("提取基礎線條與明暗。")
        
    with v2:
        st.subheader("第二階段：零件")
        # 強化對比模擬局部零件提取
        part_view = img.filter(ImageFilter.SHARPEN).convert('RGB')
        st.image(part_view, use_container_width=True)
        st.write("組合線條，認出零件形狀。")
        
    with v3:
        st.subheader("第三階段：語意")
        # 熱力圖模擬 AI 注意力分佈
        heatmap = img.convert('L').resize((14, 14)).resize(img.size, resample=Image.NEAREST)
        heatmap = ImageOps.colorize(heatmap, black="blue", white="red")
        st.image(heatmap, use_container_width=True)
        st.write("拋棄細節，理解物件的空間邏輯。")

    # --- 4. 辨識結果 ---
    st.markdown("---")
    st.header("🏆 最終判斷結果")
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(input_t)
        prob = torch.nn.functional.softmax(out[0], dim=0)
        top_p, top_id = torch.topk(prob, 1)
        st.metric(label="AI 認出的物件是：", value=labels[top_id[0]], delta=f"信心值 {top_p[0]:.2%}")

st.write("---")
st.info("💡 業界實例：在自駕車系統中，CNN 負責快速抓取路況特徵；在醫療診斷中，深層的 CNN（如 ResNet）則負責精準比對腫瘤紋理。")
