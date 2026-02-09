import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests
import numpy as np

# 頁面設定
st.set_page_config(page_title="AI 影像專家系統", layout="wide")

# --- 1. 核心大腦：載入模型 ---
@st.cache_resource
def get_resources():
    # 載入 ResNet-101 (深度與穩定代表)
    res101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    res101.eval()
    # 載入 EfficientNet-B0 (效率與精準代表)
    eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    eff_b0.eval()
    # 載入分類標籤
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
    return res101, eff_b0, labels

res101, eff_b0, labels = get_resources()

# --- 2. 介面標題 ---
st.title("🤖 AI 影像探險：從『看見線條』到『理解物件』")
st.markdown("""
這個 App 會帶你進入 **CNN (卷積神經網路)** 的隱藏層世界。
我們會對比經典的 **ResNet-101** 與現代的 **EfficientNet**，看看它們如何理解你上傳的照片。
""")

# 側邊欄：教學設定
with st.sidebar:
    st.header("🏢 實驗室設定")
    model_choice = st.radio("選擇 AI 大腦：", ["ResNet-101 (重裝深層)", "EfficientNet-B0 (輕量高效)"])
    st.markdown("---")
    st.markdown("### 業界專家小叮嚀")
    st.write("在業界，ResNet 常用於**醫療影像**，因為它架構穩定；EfficientNet 常用於**手機 App**，因為它又快又省電。")

# --- 3. 圖片上傳區 ---
uploaded_file = st.file_uploader("📸 上傳一張照片來測試 (例如貓、狗、車、花...)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    # 前處理視覺化
    col_input, col_ai_view = st.columns(2)
    with col_input:
        st.header("📥 你的原始照片")
        st.image(img, use_container_width=True)
    with col_ai_view:
        st.header("👓 AI 的初步印象")
        # 模擬 AI 前處理：縮放並中心裁剪
        ai_view = img.resize((224, 224))
        st.image(ai_view, caption="AI 實際上只看這塊 224x224 的區域", width=224)

    st.markdown("---")

    # --- 4. 隱藏層視覺化：凸顯特徵處理含意 ---
    st.header(f"🏗️ {model_choice} 的特徵提取過程")
    st.info("AI 並不是直接看到整張圖，而是在隱藏層中進行『特徵過濾』。")

    v_col1, v_col2, v_col3 = st.columns(3)

    with v_col1:
        st.subheader("1. 邊緣處理 (Edges)")
        # 視覺化模擬：使用 FIND_EDGES 模擬淺層卷積
        edge_map = img.convert('L').filter(ImageFilter.FIND_EDGES)
        st.image(edge_map, caption="淺層：偵測輪廓與線條", use_container_width=True)
        st.write("🔍 **AI 在做什麼？** 尋找物體的邊界、條紋與顏色交界處。")

    with v_col2:
        st.subheader("2. 特徵處理 (Shapes)")
        # 視覺化模擬：強化細節並稍微模糊，模擬局部特徵圖
        feature_map = img.filter(ImageFilter.DETAIL).resize((img.width // 2, img.height // 2))
        st.image(feature_map, caption="中層：認出形狀零件", use_container_width=True)
        st.write("📐 **AI 在做什麼？** 將線條組合成三角形、圓形或紋理，認出『耳朵』或『輪子』。")

    with v_col3:
        st.subheader("3. 物件處理 (Concepts)")
        # 視覺化模擬：極度像素化，模擬高階抽象權重
        concept_map = img.resize((14, 14)).resize((img.width, img.height), resample=Image.NEAREST)
        st.image(concept_map, caption="深層：理解物件語意", use_container_width=True)
        st.write("🧩 **AI 在做什麼？** 這是最抽象的階段，它在確認這些零件的空間關係，判斷『這是一隻貓』。")

    # --- 5. 辨識結果 ---
    st.markdown("---")
    st.header("🏆 辨識決策階段")
    
    # 模型運算預處理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    with st.spinner('正在穿越 101 層隱藏層...'):
        model = res101 if "ResNet" in model_choice else eff_b0
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_id = torch.topk(prob, 5)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("🎯 猜測結果")
        for i in range(5):
            st.write(f"第 {i+1} 名: **{labels[top5_id[i]]}**")
    with res_col2:
        st.subheader("📊 信心指數")
        for i in range(5):
            st.progress(float(top5_prob[i]), text=f"{top5_prob[i]:.2%}")

    # --- 6. 業界大解密：為什麼這兩個模型很強？ ---
    st.markdown("---")
    st.header("🤝 業界實戰：合併使用的藝術")
    
    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("### 🏢 ResNet-101 的強項")
        st.write("**核心：跳躍捷徑 (Skip Connection)**")
        st.write("它像是一棟結構紮實的摩天大樓。即使蓋到 101 層，只要有『快速道路』，訊息就不會出錯。業界常用於需要**絕對穩定**的場景，如工業零件檢測。")
    with exp2:
        st.markdown("### ⚡ EfficientNet 的強項")
        st.write("**核心：複合縮放 (Compound Scaling)**")
        st.write("它像是精算過後的超級跑車。不盲目追求層數，而是讓寬度與解析度達到黃金比例。業界常用於**實時辨識**，如監視器或手機 App。")

    st.success("💡 **業界趨勢：** 現在最厲害的技術會將兩者『合併』。用 ResNet 的穩定當骨幹，配上 EfficientNet 的縮放邏輯，打造出既準又快的新模型（如 ConvNeXt）！")
    
    st.balloons()
