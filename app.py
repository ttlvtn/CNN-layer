import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import requests

st.set_page_config(page_title="AI 大腦實驗室", layout="wide")

# --- 1. 核心邏輯：載入模型與標籤 ---
@st.cache_resource
def get_resources():
    # 載入 ResNet-101 (深度代表)
    res101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    res101.eval()
    # 載入 EfficientNet-B0 (效率代表)
    eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    eff_b0.eval()
    # 載入分類標籤
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
    return res101, eff_b0, labels

res101, eff_b0, labels = get_resources()

# --- 2. UI 介面設計 ---
st.title("🧠 AI 大腦實驗室：ResNet vs EfficientNet")
st.markdown("""
本實驗室將展示兩大經典 AI 架構如何處理影像。你可以觀察「深度」與「效率」在視覺化處理上的差異。
""")

with st.sidebar:
    st.header("工廠大腦設定")
    model_choice = st.radio("你想用哪個模型？", ["ResNet-101 (深度取勝)", "EfficientNet-B0 (效率取勝)"])
    st.markdown("---")
    st.info("💡 **小知識：** 業界現在常將兩者結合，ResNet 負責穩定的基礎，EfficientNet 負責優化效率。")

# --- 3. 步驟一：上傳與前處理視覺化 ---
uploaded_file = st.file_uploader("請上傳一張圖片...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    col_input, col_process = st.columns(2)
    with col_input:
        st.header("🖼️ 原始輸入")
        st.image(img, use_container_width=True)
        
    with col_process:
        st.header("🔍 AI 的前處理視野")
        ai_view = img.resize((224, 224))
        st.image(ai_view, caption="圖片會被強制調整為 224x224 供 AI 讀取", width=224)
        st.write("1. 尺寸統一 2. 顏色標準化 3. 轉化為數字矩陣")

    st.markdown("---")

    # --- 4. 步驟二：加工過程模擬視覺化 (修正 Bug) ---
    st.header(f"🏗️ {model_choice} 的加工過程模擬")
    st.write("AI 並不是一眼看出答案，而是透過隱藏層一層層「抽絲剝繭」。")
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.image(img, caption="第一階段：邊緣偵測", use_container_width=True)
        st.caption("🔍 提取基礎特徵：線條、顏色對比。")

    with v_col2:
        # 修復之處：使用 ImageFilter.BoxBlur 並確保大小寫正確
        img_mid = img.resize((img.width // 2, img.height // 2)).filter(ImageFilter.BoxBlur(radius=2))
        st.image(img_mid, caption="第二階段：特徵組合", use_container_width=True)
        st.caption("📐 辨識局部形狀：如耳朵、輪胎弧度。")

    with v_col3:
        img_deep = img.resize((img.width // 4, img.height // 4)).filter(ImageFilter.BoxBlur(radius=4))
        st.image(img_deep, caption="第三階段：物件特徵", use_container_width=True)
        st.caption("🧩 抽象化理解：確認這是一個完整的物件。")

    # --- 5. 步驟三：運作邏輯圖解 ---
    logic_col1, logic_col2 = st.columns(2)
    if "ResNet" in model_choice:
        with logic_col1:
            st.subheader("ResNet 邏輯：跳躍捷徑")
            st.write("像是有 101 個加工區，並設有「快速道路」。")
            st.write("即使工廠再深，資訊也不會迷失。")
        with logic_col2:
            st.graphviz_chart('''
            digraph { rankdir=LR; node[shape=box, style=filled, color=lightblue]; 
            Input -> Layer1 -> Layer2 -> Layer3 -> Output;
            Layer1 -> Layer3 [label="捷徑 (Skip)", color=red]; }
            ''')
            
    else:
        with logic_col1:
            st.subheader("EfficientNet 邏輯：複合縮放")
            st.write("不一味加深，而是精算深度、寬度與解析度。")
            st.write("用數學找到最省電且最精準的黃金比例。")
        with logic_col2:
            st.latex(r"Scaling = (\text{depth}, \text{width}, \text{res})")
            

    # --- 6. 步驟四：辨識結果 ---
    st.markdown("---")
    st.header("🏆 辨識結果")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    with st.spinner('AI 正在 101 層隱藏層中旅行...'):
        current_model = res101 if "ResNet" in model_choice else eff_b0
        with torch.no_grad():
            output = current_model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_id = torch.topk(prob, 5)

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        for i in range(5):
            st.write(f"排名 {i+1}: **{labels[top5_id[i]]}**")
    with res_col2:
        for i in range(5):
            st.progress(float(top5_prob[i]), text=f"信心度：{top5_prob[i]:.2%}")

    # --- 7. 業界實例對比表 ---
    st.markdown("---")
    st.header("🏢 業界實例：它們如何合作？")
    st.table({
        "場景": ["醫院 X 光篩檢", "手機拍照美顏", "自駕車物件辨識"],
        "架構選擇": ["ResNet + EfficientNet 集成", "輕量化 EfficientNet", "ResNet 骨幹 + 自定義層"],
        "原因": ["醫療不能出錯，多個模型投票更穩", "需要省電、即時反應", "需要深層特徵來確保安全"]
    })

    st.balloons()
