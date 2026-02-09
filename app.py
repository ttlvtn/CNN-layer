import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests

st.set_page_config(page_title="AI 大腦實驗室", layout="wide")

# --- 1. 載入模型 ---
@st.cache_resource
def get_resources():
    res101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    res101.eval()
    eff_b0.eval()
    
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
    return res101, eff_b0, labels

res101, eff_b0, labels = get_resources()

# --- 2. UI 標題 ---
st.title("🧠 AI 大腦實驗室：ResNet vs EfficientNet")
st.markdown("想要知道肌肉男 (ResNet) 和 智慧大師 (EfficientNet) 誰比較準嗎？")

# 側邊欄切換
with st.sidebar:
    st.header("工廠設定")
    model_choice = st.radio("選擇辨識大腦：", ["ResNet-101", "EfficientNet-B0"])
    st.markdown("---")
    st.info("💡 業界小知識：現在很多最強的 AI 其實是這兩者的『混血兒』喔！")

# --- 3. 圖片上傳 ---
uploaded_file = st.file_uploader("上傳一張圖片，開始實驗...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="原始圖片", use_container_width=True)

    st.markdown("---")
    st.header(f"⚙️ {model_choice} 的加工過程視覺化")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    
    # 視覺化模擬 (修正後的濾鏡寫法)
    with col_v1:
        st.image(img, caption="1. 淺層：找邊緣", use_container_width=True)
        st.write("🔍 正在偵測線條與顏色...")

    with col_v2:
        # 使用 ImageFilter.BoxBlur 修正 Bug
        img_mid = img.resize((img.width // 2, img.height // 2)).filter(ImageFilter.BoxBlur(radius=2))
        st.image(img_mid, caption="2. 中層：找形狀", use_container_width=True)
        st.write("📐 正在組合成耳朵、輪胎等形狀...")

    with col_v3:
        img_deep = img.resize((img.width // 4, img.height // 4)).filter(ImageFilter.BoxBlur(radius=4))
        st.image(img_deep, caption="3. 深層：認物件", use_container_width=True)
        st.write("🧩 最終確認這是一個什麼東西！")

    # --- 4. 辨識結果 ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    with st.spinner('AI 腦袋飛速運轉中...'):
        model = res101 if model_choice == "ResNet-101" else eff_b0
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_id = torch.topk(prob, 1)

    st.success(f"🏆 辨識結果：**{labels[top_id[0]]}** (信心指數：{top_prob[0]:.2%})")

    # --- 5. 業界實例區 (整合進 App) ---
    st.markdown("---")
    st.header("🏢 業界實戰：它們都用在哪？")
    
    tab1, tab2 = st.tabs(["醫療與工業", "手機與生活"])
    
    with tab1:
        st.subheader("🏥 醫院的 X 光自動診斷")
        st.write("**方案：** 通常合併使用 ResNet + EfficientNet。")
        st.write("**理由：** 醫療不能出錯。ResNet 看細節，EfficientNet 看結構，兩個都點頭醫師才放心。")
    
    with tab2:
        st.subheader("📸 手機的人像模式")
        st.write("**方案：** 使用高效能的 EfficientNet 變種。")
        st.write("**理由：** 手機拍照不能發燙，也不能讓使用者等太久，所以『效率』是第一優先！")

st.balloons()
