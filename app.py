import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io # For handling image bytes

st.set_page_config(page_title="AI 探險隊：ResNet 深度工廠", layout="wide")

# --- 載入模型和預處理轉換 ---
@st.cache_resource # 讓模型只載入一次
def load_resnet_model():
    # 載入一個已經在 ImageNet (一百萬張圖) 練好的 ResNet-101
    resnet101 = models.resnet101(pretrained=True)
    resnet101.eval() # 設定為「預測模式」
    return resnet101

@st.cache_resource # 載入 ImageNet 的分類標籤
def load_imagenet_labels():
    # 從網路下載 ImageNet 1000 個類別的標籤
    import requests
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
    return labels

resnet_model = load_resnet_model()
imagenet_labels = load_imagenet_labels()

# 圖片預處理轉換
preprocess = transforms.Compose([
    transforms.Resize(256),          # 圖片縮放到 256x256
    transforms.CenterCrop(224),      # 從中心裁剪出 224x224
    transforms.ToTensor(),           # 轉換成 PyTorch Tensor 格式 (HWC -> CHW, 0-255 -> 0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 標準化處理
])

# --- UI 介面 ---
st.title("🚀 AI 探險隊：解密 ResNet-101 深度工廠")
st.subheader("上傳圖片，讓 ResNet 101 帶你深入了解它的『思考』過程！")

st.markdown("""
---
👋 **給同學們：** 想像 ResNet-101 是一間超級聰明的影像辨識工廠。
你上傳一張圖片，它會經過 **101 層** 的精密加工，最後告訴你圖片裡有什麼！
""")

# --- 圖片上傳區 ---
st.header("📸 步驟一：上傳你的圖片")
uploaded_file = st.file_uploader("選擇一張圖片 (建議上傳貓、狗、鳥、車子等常見物品的圖片)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片並顯示
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="你上傳的圖片", use_column_width=True)

    st.markdown("---")
    st.header("⚙️ 步驟二：圖片進入工廠前處理")
    st.info("你的圖片不是直接送進 AI 腦袋喔！它需要先被『翻譯』成 AI 懂的語言。")
    st.write("""
    1.  **尺寸統一：** 圖片會被縮小到 AI 規定的尺寸（例如：224x224 像素）。
    2.  **顏色標準化：** 圖片的紅綠藍 (RGB) 數值會被調整，讓 AI 不會被圖片的亮度或對比度誤導。
    3.  **轉換格式：** 變成 AI 讀得懂的數字矩陣。
    """)

    # 顯示預處理後的結果 (這裡只是文字說明，實際不顯示圖片變化)
    st.markdown("---")
    st.header("🧠 步驟三：深入 ResNet-101 隱藏層的『思考』")
    st.markdown("現在，你的圖片開始在 101 層隱藏層中旅行了！")

    st.write("---")
    st.subheader("加工流程：")
    
    col_proc1, col_proc2, col_proc3, col_proc4 = st.columns(4)

    with col_proc1:
        st.success("1. 卷積 (Convolution) 🔍")
        st.write("工廠的『偵探』：用濾鏡掃描圖片，找出局部特徵（線條、顏色、紋理）。")
        st.write("就像在找『這是貓毛的紋路』、『這是車子的輪廓線』。")

    with col_proc2:
        st.success("2. 標準化 (Batch Norm) 📏")
        st.write("工廠的『秩序維護者』：把偵測到的特徵數值排整齊，避免數值過大過小影響效率。")
        st.write("確保 AI 大腦不會『過度興奮』或『完全沒反應』。")

    with col_proc3:
        st.success("3. 激活 (ReLU) 💡")
        st.write("工廠的『決策者』：重要的特徵（正數）留下，不重要的（負數）捨棄變零。")
        st.write("決定哪些訊息可以傳遞給下一層。")
        
    with col_proc4:
        st.warning("🔄 ResNet 的獨家密技：跳躍捷徑！")
        st.write("如果某層加工員不給力，資訊可以直接走捷徑到下一層。")
        st.write("這樣即使網路很深，也不會『迷路』，能一直高效學習。")
        st.caption("這個獨家密技貫穿了 ResNet-101 的所有隱藏層！")

    st.markdown("---")
    st.header("🏆 步驟四：ResNet-101 的最終判斷！")
    
    # 執行預處理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # 增加一個維度以符合模型輸入要求 (Batch_size x Channels x Height x Width)

    # 執行模型預測
    with torch.no_grad(): # 預測時不需要計算梯度
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
