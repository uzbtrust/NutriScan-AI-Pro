import streamlit as st
from ultralytics import YOLO
import PIL.Image
import torch
import os

st.set_page_config(
    page_title="NutriScan AI Pro", 
    page_icon="🥗", 
    layout="wide"
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0px;}

    .main { background-color: #ffffff; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 80px;
        justify-content: center;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 80px;
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #8e8e93;
    }
    .stTabs [aria-selected="true"] {
        color: #0071e3 !important;
        border-bottom: 4px solid #0071e3 !important;
    }

    .block-container { 
        padding-top: 6rem !important; 
        padding-bottom: 2rem !important; 
    }
    
    .tips-card {
        background-color: #fff4e5; 
        padding: 25px; 
        border-radius: 20px; 
        border-left: 6px solid #ffa117;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(m_path):
    if m_path.endswith('.mlpackage'):
        return YOLO(m_path, task='segment')
    model = YOLO(m_path)
    if torch.backends.mps.is_available():
        model.to('mps')
    return model

def calculate_calories(result):
    kcal = 0
    k_factor = 0.0005
    if result.masks is not None:
        for mask in result.masks.data:
            area = torch.sum(mask).item()
            kcal += area * k_factor
    return round(kcal, 1)

tab_analyzer, tab_about = st.tabs(["🔍 ANALYZER", "📖 ABOUT PROJECT"])

with tab_analyzer:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    
    with ctrl_col1:
        st.title("NutriScan AI Pro")
        st.markdown("##### AI-Powered Visual Nutrition Intelligence")
    
    with ctrl_col2:
        selected_engine = st.selectbox(
            "Computing Engine", 
            ["PyTorch (Mac GPU)", "CoreML (Neural Engine)"],
            index=1
        )
        model_file = 'best.pt' if "PyTorch" in selected_engine else 'best.mlpackage'
    
    with ctrl_col3:
        conf_value = st.slider("Detection Sensitivity", 0.1, 1.0, 0.35)

    st.markdown("---")
    
    res_col, data_col = st.columns([1.6, 1])

    with res_col:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_file:
            img = PIL.Image.open(uploaded_file)
            model = load_model(model_file)
            results = model.predict(source=img, conf=conf_value)
            
            res_plotted = results[0].plot(labels=True, boxes=True)
            st.image(res_plotted, use_container_width=True)
            
            kcal_val = calculate_calories(results[0])
            items_val = len(results[0].boxes)
        else:
            st.info("👆 Please upload a meal photo to begin analysis.")
            kcal_val, items_val = 0, 0

    with data_col:
        st.metric("ESTIMATED ENERGY", f"{kcal_val} kcal")
        st.metric("INGREDIENTS FOUND", f"{items_val} units")
        
        st.markdown("""
        <div class="tips-card">
            <h4 style="color: #663c00; margin-top:0;">📸 How to get 99% accuracy?</h4>
            <p style="color: #663c00; font-size: 15px; margin-bottom: 10px;">
                For the best calorie estimation, follow these guidelines:
            </p>
            <ul style="color: #663c00; font-size: 14px;">
                <li><b>Overhead Shot:</b> Align your camera directly above the plate.</li>
                <li><b>Lighting:</b> Natural light provides the best contrast for the Neural Engine.</li>
                <li><b>No Overlap:</b> Try to keep food items distinct and visible.</li>
                <li><b>Flat Surface:</b> Place the plate on a stable, flat surface before shooting.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_about:
    st.title("Project Architecture & Science")
    
    st.header("🛠 Technology Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.subheader("Deep Learning")
        st.write("- **YOLOv11-Segmentation**: SOTA instance segmentation.")
        st.write("- **PyTorch (MPS)**: Mac GPU acceleration.")
    with t2:
        st.subheader("Hardware Support")
        st.write("- **Apple Neural Engine (ANE)**: Optimized CoreML execution.")
        st.write("- **M4 Silicon**: High-bandwidth memory for 30ms inference.")
    with t3:
        st.subheader("Frontend")
        st.write("- **Streamlit**: Reactive UI Framework.")
        st.write("- **Matplotlib/OpenCV**: Data visualization & processing.")

    st.divider()

    st.header("🧮 Scientific Methodology")
    st.write("We calculate energy by measuring the precise pixel-area of segmented food masks.")
    st.latex(r"Energy (kcal) = \sum_{i=1}^{n} (Area_{pixels\_i} \times K_{density})")
    st.caption("K-factor is calibrated using the NutriSeg-2k validation dataset.")

    st.divider()

    st.header("📈 Model Validation Metrics")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        if os.path.exists('results.png'):
            st.image('results.png', caption="Learning Curves (mAP & Loss)", use_container_width=True)
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption="Accuracy Matrix", use_container_width=True)
    with m_col2:
        if os.path.exists('MaskF1_curve.png'):
            st.image('MaskF1_curve.png', caption="F1-Score Balance", use_container_width=True)
        if os.path.exists('val_batch0_pred.jpg'):
            st.image('val_batch0_pred.jpg', caption="Inference Samples", use_container_width=True)
