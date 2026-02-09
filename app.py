import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import imageio.v2 as imageio # For GIF creation if needed, though we'll simplify with static images
import numpy as np

st.set_page_config(page_title="AI 探險隊：ResNet 深度工廠", layout="wide")

# --- 載入模型和預處理轉換 ---
@st.cache_resource
def load_resnet_model():
    resnet101 = models.resnet101(pretrained=True)
    resnet101.eval()
    return resnet101

@st.cache_resource
def load_imagenet_labels():
    import requests
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
    return labels

resnet_model = load_resnet_model()
imagenet_labels = load_imagenet_labels()

# 圖片預處理轉換 (針對模型輸入)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 圖片縮放過程的轉換 (用於視覺化)
visual_resizes = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

# --- UI 介面 ---
st.title("🚀 AI 探險隊：解密 ResNet-101 深度工廠")
st.subheader("上傳圖片，讓 ResNet 101 帶你深入了解它的『思考』過程與『視覺』變化！")

st.markdown("""
---
👋 **給同學們：** 想像 ResNet-101 是一間超級聰明的影像辨識工廠。
你上傳一張圖片，它會經過 **101 層** 的精密加工，每層都讓圖片的資訊『變形』，最後告訴你圖片裡有什麼！
""")

# --- 圖片上傳區 ---
st.header("📸 步驟一：上傳你的圖片")
uploaded_file = st.file_uploader("選擇一張圖片 (建議上傳貓、狗、鳥、車子等常見物品的圖片)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # 顯示原始圖片
    st.image(original_image, caption="你上傳的原始圖片", use_column_width=True)

    st.markdown("---")
    st.header("⚙️ 步驟二：圖片進入工廠前處理 (尺寸與顏色轉換)")
    st.info("你的圖片不是直接送進 AI 腦袋喔！它需要先被『翻譯』成 AI 懂的語言。")
    
    col_pre_img, col_pre_desc = st.columns(2)
    with col_pre_img:
        # 顯示經過大小轉換的圖片
        processed_img_for_display = visual_resizes(original_image)
        st.image(processed_img_for_display, caption="AI 處理後的標準尺寸 (例如 224x224 像素)", width=224)
    with col_pre_desc:
        st.write("""
        1.  **尺寸統一：** 圖片會被縮小並裁剪到 AI 規定的標準尺寸（例如：224x224 像素），這樣 AI 不會因為圖片大小不同而困擾。
        2.  **顏色標準化：** 圖片的紅綠藍 (RGB) 數值會被調整，讓 AI 不會被圖片的亮度或對比度誤導，專注在內容上。
        3.  **轉換格式：** 最後，圖片會變成 AI 讀得懂的數字矩陣（一大堆數字）。
        """)

    st.markdown("---")
    st.header("🧠 步驟三：深入 ResNet-101 隱藏層的『思考』與『視覺變化』")
    st.markdown("現在，你的圖片開始在 101 層隱藏層中旅行了！它會被層層『加工』，每一次加工都會改變圖片的『樣子』！")

    st.subheader("核心加工區的『標準動作』與視覺變化：")
    
    # 模擬視覺化卷積層的變化 (簡化為抽象說明和縮小/模糊效果)
    st.info("💡 **模擬變化：** 由於真實的內部變化複雜且難以直接展示，這裡用漸進式的『抽象化』效果來模擬。")
    
    # 創建一些模擬中間圖
    col_stage1, col_stage2, col_stage3 = st.columns(3)
    
    # 模擬原始圖經過初步卷積，提取出邊緣特徵
    with col_stage1:
        st.image(original_image, caption="原始圖片", use_column_width=True)
        st.markdown("**➡️ 經過第一層捲積...**")
        st.write("🔍 **捲積：** 圖片開始被濾鏡掃描，抓取最基礎的**邊緣和顏色塊**。它看到了『這是亮色與暗色的交界』。")
        # 這裡不顯示真實卷積圖，而是用文字說明
        st.markdown("**(圖片開始被AI『抽象化』)**")

    # 模擬經過中間層卷積，提取出形狀特徵
    with col_stage2:
        # 稍微縮小並模糊化，模擬資訊被濃縮的感覺
        img_mid_stage = original_image.resize((original_image.width//2, original_image.height//2)).filter(Image.boxblur(radius=2))
        st.image(img_mid_stage, caption="多層加工後（特徵濃縮）", use_column_width=True)
        st.markdown("**➡️ 經過中間層捲積與池化...**")
        st.write("📏 **池化：** 圖片尺寸逐漸縮小，只留下最重要的訊息，就像『濃縮』了一樣。")
        st.write("💡 **激活：** 不重要的訊息被丟掉，AI 聚焦在關鍵特徵上。")
        st.write("🔍 **捲積：** AI 開始組合邊緣，看見『這是一隻耳朵的形狀』、『這是輪胎的圓形』。")
        st.markdown("**(圖片資訊被 AI 理解成『形狀』)**")


    # 模擬經過深層卷積，提取出高階特徵
    with col_stage3:
        # 更進一步縮小並模糊，或甚至用抽象圖塊模擬
        img_deep_stage = original_image.resize((original_image.width//4, original_image.height//4)).filter(Image.boxblur(radius=4))
        st.image(img_deep_stage, caption="深層加工後（抽象特徵）", use_column_width=True)
        st.markdown("**➡️ 經過深層捲積與跳躍捷徑...**")
        st.write("🔄 **跳躍捷徑：** 避免資訊遺失，讓深層網路也能學到東西。")
        st.write("🔍 **捲積：** AI 將這些形狀組合起來，最終認出『這是一隻貓的臉』或『這是一台車子的側面』。")
        st.markdown("**(圖片資訊被 AI 理解成『物件』)**")

    st.markdown("---")
    st.subheader("✨ ResNet-101 的獨家密技：『跳躍捷徑』視覺化")
    st.write("想像資料在工廠裡走，如果某個加工區沒辦法幫忙，它可以直接走**旁邊的捷徑**到下一關！")
    
    # 簡化版的捷徑視覺化 (可替換為更精美的圖片或 GIF)
    st.graphviz_chart('''
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue];
        subgraph cluster_main {
            label="隱藏層加工區";
            "輸入訊號" -> "卷積+標準化+激活" -> "輸出訊號";
            "輸入訊號" -> "輸出訊號" [label="跳躍捷徑", color=red, style=dashed];
        }
    }
    ''')
    st.success("這條捷徑，讓 ResNet 網路即使深達 101 層，也能高效學習，不會『資訊迷路』！")


    st.markdown("---")
    st.header("🏆 步驟四：ResNet-101 的最終判斷！")
    
    # 執行預處理 (使用為模型準備的轉換)
    input_tensor = preprocess(original_image)
    input_batch = input_tensor.unsqueeze(0) # 增加一個 Batch 維度

    # 執行模型預測
    with torch.no_grad():
        output = resnet_model(input_batch)

    # 取得前 5 個最可能的預測結果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.success("✨ **ResNet-101 工廠經過 101 層加工後，判斷這張圖最可能是：**")
    for i in range(top5_prob.size(0)):
        st.write(f"**{i+1}. {imagenet_labels[top5_catid[i]]}** (信心指數：{top5_prob[i].item():.2%})")

    st.info("每個分類後面括號裡的數字，代表 AI 覺得這個答案有多大的可能性！")

st.markdown("---")
st.write("💡 **課後小思考：** 如果 AI 猜錯了，可能是什麼原因呢？(提示：AI 沒看過這種圖、圖片模糊、AI 沒學過這個東西)")
