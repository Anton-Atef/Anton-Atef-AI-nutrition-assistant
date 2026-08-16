import os, io, re, json, math, random
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="Live AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- CUSTOM CSS ---------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); }
.food-card {
    background: white; border-radius: 20px; padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom:15px;
}
.macro-badge {
    display:inline-block; padding:8px 14px; border-radius:12px;
    font-weight:600; font-size:14px; margin:5px;
}
</style>
""", unsafe_allow_html=True)

# --------- CONSTANTS ---------
CFG_IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FOOD_DB = {
    "salad": {"calories": 35, "protein": 2.5, "fat": 0.8, "carb": 6, "fiber": 2.5, "sugar": 2.2, "name": "Vegetable Salad"},
    "spaghetti": {"calories": 158, "protein": 5.8, "fat": 0.9, "carb": 30.9, "fiber": 1.8, "sugar": 0.6, "name": "Spicy Tomato Fusilli"},
    "sushi": {"calories": 130, "protein": 6, "fat": 1.5, "carb": 22, "fiber": 0.8, "sugar": 3, "name": "Salmon Bowl"},
    "pizza": {"calories": 266, "protein": 11, "fat": 10, "carb": 33, "fiber": 2.3, "sugar": 3.6, "name": "Pizza Slice"},
    "burger": {"calories": 295, "protein": 17, "fat": 14, "carb": 24, "fiber": 1.5, "sugar": 4, "name": "Chicken Burger"},
    "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carb": 14, "fiber": 2.4, "sugar": 10, "name": "Apple"},
    "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carb": 23, "fiber": 2.6, "sugar": 12, "name": "Banana"},
    "chicken breast": {"calories": 165, "protein": 31, "fat": 3.6, "carb": 0, "fiber": 0, "sugar": 0, "name": "Chicken Breast"},
    "rice": {"calories": 130, "protein": 2.4, "fat": 0.3, "carb": 28, "fiber": 0.4, "sugar": 0.05, "name": "Rice"},
    "salmon": {"calories": 208, "protein": 20, "fat": 13, "carb": 0, "fiber": 0, "sugar": 0, "name": "Salmon"},
    "yogurt": {"calories": 59, "protein": 3.5, "fat": 0.4, "carb": 5, "fiber": 0, "sugar": 3.2, "name": "Greek Yogurt"},
    "cake": {"calories": 371, "protein": 5, "fat": 16, "carb": 52, "fiber": 0.8, "sugar": 31, "name": "Chocolate Cake"},
}

# --------- SESSION STATE ---------
defaults = {
    "meals": [], "shopping_list": [], "chat_history": [],
    "user_profile": {"age":25,"sex":"Male","weight":70,"height":175,"activity":"Moderate","goal":"Maintenance"},
    "daily_target": None,
    "ocr_history": []
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------- HELPERS - NUTRITION MATH ---------
def calculate_bmr(w,h,age,sex):
    if sex=="Male": return 88.362 + (13.397*w) + (4.799*h) - (5.677*age)
    return 447.593 + (9.247*w) + (3.098*h) - (4.330*age)

def calculate_tdee(bmr, activity):
    mult = {"Sedentary":1.2,"Light":1.375,"Moderate":1.55,"Active":1.725,"Very Active":1.9}
    return bmr * mult.get(activity,1.55)

def calculate_goals(tdee, weight, goal):
    if goal=="Weight Loss": cal=tdee-500
    elif goal=="Muscle Gain": cal=tdee+300
    else: cal=tdee
    if goal=="Muscle Gain": p = weight*2.2
    elif goal=="Weight Loss": p = weight*2.0
    else: p = weight*1.8
    f = (cal*0.25)/9
    c = (cal - (p*4 + f*9))/4
    return {"calories": round(cal), "protein": round(p), "fat": round(f), "carb": round(c), "fiber": 25}

def get_today_meals():
    today = date.today().isoformat()
    return [m for m in st.session_state.meals if m.get("date")==today]

def get_totals():
    meals = get_today_meals()
    tot = {"calories":0,"protein":0,"fat":0,"carb":0,"fiber":0}
    for m in meals:
        n=m["nutrition"]
        for k in tot: tot[k]+=n.get(k,0)
    return tot

# --------- MODEL DEFINITIONS (for your checkpoint) ---------
try:
    import torch
    import torch.nn as nn
    import timm
    HAS_TORCH=True
except:
    HAS_TORCH=False
    torch=None

if HAS_TORCH:
    class NutritionNet(nn.Module):
        def __init__(self, backbone_name="convnext_small", in_chans=4, out_dim=5):
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=False, in_chans=in_chans, num_classes=0, global_pool="avg")
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 512),
                nn.GELU(), nn.Dropout(0.20), nn.Linear(512, out_dim)
            )
        def forward(self, x):
            return self.head(self.backbone(x))

    @st.cache_resource
    def load_nutrition_model():
        from huggingface_hub import hf_hub_download
        try:
            # Download the 200MB checkpoint from your HF model repo (cached after first run)
            ckpt_path = hf_hub_download(
                repo_id="Anton-Atef/AI-nutrition-assistant",
                filename="best_nutrition_rgbd.pt",
                repo_type="model"   # it's in a MODEL repo, not the Space
            )
        except Exception as e:
            st.warning(f"Could not download Nutrition5k checkpoint: {e}")
            return None, None, None

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
            y_std = np.array(ckpt["y_std"], dtype=np.float32)
            cols = ckpt.get("target_cols", ["total_mass","total_calories","total_fat","total_carb","total_protein"])
            model = NutritionNet(ckpt.get("backbone","convnext_small"), in_chans=ckpt.get("in_chans",4), out_dim=len(cols))
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model, (y_mean, y_std), cols
        except Exception as e:
            st.warning(f"Could not load Nutrition5k checkpoint: {e}")
            return None, None, None
else:
    def load_nutrition_model(): return None, None, None

@st.cache_resource
def load_food_classifier():
    try:
        from transformers import pipeline
        # lightweight food model, will cache
        pipe = pipeline("image-classification", model="nateraw/food", top_k=3)
        return pipe
    except Exception as e:
        return None

@st.cache_resource
def load_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        return None

# --------- OCR PARSER ---------
def parse_nutrition_label_text(text: str) -> Dict:
    t = text.lower()
    def find(pattern):
        m = re.search(pattern, t, re.I)
        if m: 
            try: return float(m.group(1))
            except: return None
        return None
    out = {
        "calories": find(r"calories?\s*[:]*\s*(\d+\.?\d*)"),
        "total_fat_g": find(r"total fat[^\d]*(\d+\.?\d*)\s*g"),
        "saturated_fat_g": find(r"saturated fat[^\d]*(\d+\.?\d*)"),
        "trans_fat_g": find(r"trans fat[^\d]*(\d+\.?\d*)"),
        "cholesterol_mg": find(r"cholesterol[^\d]*(\d+\.?\d*)\s*mg"),
        "sodium_mg": find(r"sodium[^\d]*(\d+\.?\d*)\s*mg"),
        "total_carbohydrates_g": find(r"(?:total\s*)?carbohydrate[^\d]*(\d+\.?\d*)\s*g"),
        "dietary_fiber_g": find(r"dietary fiber[^\d]*(\d+\.?\d*)\s*g"),
        "total_sugars_g": find(r"total sugars?[^\d]*(\d+\.?\d*)\s*g"),
        "added_sugars_g": find(r"added sugars?[^\d]*(\d+\.?\d*)\s*g"),
        "protein_g": find(r"protein[^\d]*(\d+\.?\d*)\s*g"),
        "serving_size": None,
        "raw_text": text[:500]
    }
    return out

def estimate_from_food_db(label: str, grams: float):
    label = label.lower()
    best = None
    for k,v in FOOD_DB.items():
        if k in label:
            best=v
            break
    if not best: best = FOOD_DB["salad"]
    factor = grams/100
    return {
        "name": best["name"],
        "calories": round(best["calories"]*factor,1),
        "protein": round(best["protein"]*factor,1),
        "fat": round(best["fat"]*factor,1),
        "carb": round(best["carb"]*factor,1),
        "fiber": round(best.get("fiber",0)*factor,1),
        "sugar": round(best.get("sugar",0)*factor,1),
        "mass": grams
    }

def predict_with_rgbd_model(pil_img, model_data):
    if model_data[0] is None: return None
    model, (y_mean, y_std), cols = model_data
    try:
        img = pil_img.resize((CFG_IMG_SIZE, CFG_IMG_SIZE))
        rgb = np.array(img).astype(np.float32)/255.0
        rgb = (rgb - IMAGENET_MEAN)/IMAGENET_STD
        depth = np.zeros((CFG_IMG_SIZE, CFG_IMG_SIZE,1), dtype=np.float32)
        depth = (depth - 0.5)/0.25
        img4 = np.concatenate([rgb, depth], axis=2)
        x = torch.from_numpy(img4).permute(2,0,1).unsqueeze(0).float()
        with torch.no_grad():
            pred = model(x).numpy()[0]
        pred = pred * y_std + y_mean
        # cols = mass, calories, fat, carb, protein
        mapping = {c:v for c,v in zip(cols,pred)}
        return {
            "name": "AI Estimated Dish",
            "mass": float(mapping.get("total_mass",250)),
            "calories": float(mapping.get("total_calories",200)),
            "fat": float(mapping.get("total_fat",10)),
            "carb": float(mapping.get("total_carb",18)),
            "protein": float(mapping.get("total_protein",12)),
            "fiber": 2.0, "sugar": 3.0
        }
    except Exception as e:
        st.error(f"RGBD inference failed: {e}")
        return None

# --------- ROUTER FROM YOUR CODE ---------
@dataclass
class Route:
    intent: str
    tool: Optional[str]
    confidence: float
    arguments: Dict[str, Any] = field(default_factory=dict)

class AIRouter:
    def __init__(self):
        self.rules = {
            "daily_summary": [r"how many calories.*(left|remaining)", r"calories.*(left|remaining)", r"what did i eat today", r"daily.*summary"],
            "remaining_macros": [r"how much protein.*(left|remaining)", r"remaining.*(protein|carb|fat|macro)"],
            "portion_calculation": [r"how much.*can i eat", r"how many grams", r"what portion"],
            "food_logging": [r"i ate", r"log this", r"add this"],
            "recommendation": [r"what should i eat", r"recommend.*meal", r"what.*eat.*dinner"],
            "food_comparison": [r"compare", r"which.*better"],
            "user_targets": [r"my calorie target", r"my macros", r"what.*my.*target"],
            "food_history": [r"food history", r"what.*ate.*yesterday", r"show.*my meals"],
        }
        self.tool_map = {
            "daily_summary": "get_daily_summary", "remaining_macros": "get_remaining_macros",
            "portion_calculation": "calculate_food_portion", "food_logging": "log_food",
            "recommendation": "recommend_meals", "food_comparison": "compare_foods",
            "user_targets": "get_user_targets", "food_history": "get_food_history",
        }
    def route(self, message: str) -> Route:
        text = re.sub(r"\s+", " ", message.lower().strip())
        scores = {intent: sum(bool(re.search(p, text)) for p in pats) for intent,pats in self.rules.items()}
        best = max(scores, key=scores.get)
        if scores[best]==0:
            return Route("general_nutrition_chat", None, 0.25)
        return Route(best, self.tool_map[best], min(0.95,0.55+scores[best]*0.15))

router = AIRouter()

def generate_coach_response(user_msg):
    totals = get_totals()
    target = st.session_state.daily_target or calculate_goals(2000,70,"Maintenance")
    remaining = {k: target[k]-totals.get(k,0) for k in ["calories","protein","carb","fat"]}
    route = router.route(user_msg)
    
    ctx = f"Today eaten: {totals}, Target: {target}, Remaining: {remaining}, Meals: {len(get_today_meals())}"
    
    # Tool-based deterministic answers
    if route.intent=="daily_summary":
        return f"**Today's Summary** 📊\n\nYou have eaten {len(get_today_meals())} meals.\n\n- Calories: {totals['calories']:.0f}/{target['calories']} (remaining {remaining['calories']:.0f})\n- Protein: {totals['protein']:.0f}g / {target['protein']}g (remaining {remaining['protein']:.0f}g)\n- Carbs: {totals['carb']:.0f}g / {target['carb']}g\n- Fat: {totals['fat']:.0f}g / {target['fat']}g\n\nYou're doing great! Keep it balanced."
    if route.intent=="remaining_macros":
        return f"You have **{remaining['calories']:.0f} kcal** left today.\n\n- Protein: {remaining['protein']:.0f}g\n- Carbs: {remaining['carb']:.0f}g\n- Fat: {remaining['fat']:.0f}g\n\nWant a meal suggestion to fill it?"
    if route.intent=="recommendation":
        if remaining['protein']>20:
            return f"With {remaining['calories']:.0f} kcal left and {remaining['protein']:.0f}g protein needed, try:\n- **Grilled Chicken + Rice + Salad** (~400 kcal, 35g protein)\n- Greek Yogurt with fruits (~150 kcal)\n- Salmon Bowl (~350 kcal, 22g protein)"
        else:
            return f"Light option with {remaining['calories']:.0f} kcal left: **Vegetable Salad + Olive oil** or **Apple + Peanut Butter**."
    if route.intent=="portion_calculation":
        return f"Tell me the food and I can calculate portion! You have {remaining['calories']:.0f} kcal remaining. For example, chocolate cake is ~371 kcal/100g, so you could have about {max(0, remaining['calories']/371*100):.0f}g."
    if route.intent=="user_targets":
        return f"Your daily targets ({st.session_state.user_profile['goal']}):\n- {target['calories']} kcal\n- Protein {target['protein']}g\n- Carbs {target['carb']}g\n- Fat {target['fat']}g"
    
    # General LLM fallback
    # Check for OpenAI key in secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":f"You are an AI Nutrition Coach. Data: {ctx}. Be concise, friendly, no shaming."},
                    {"role":"user","content":user_msg}
                ], max_tokens=300
            )
            return resp.choices[0].message.content
    except: pass
    
    return f"I'm your AI Nutrition Coach 🥗\n\n{ctx}\n\nYou asked: '{user_msg}'\n\nTips: Focus on whole foods, balance your remaining macros, and enjoy food without guilt. What would you like to log or know about your meals?"

# --------- SIDEBAR ---------
with st.sidebar:
    st.title("👤 Your Profile")
    p = st.session_state.user_profile
    p["age"] = st.number_input("Age", 10, 90, p["age"])
    p["sex"] = st.selectbox("Sex", ["Male","Female"], index=0 if p["sex"]=="Male" else 1)
    p["weight"] = st.number_input("Weight (kg)", 30.0, 200.0, float(p["weight"]))
    p["height"] = st.number_input("Height (cm)", 100.0, 230.0, float(p["height"]))
    p["activity"] = st.selectbox("Activity", ["Sedentary","Light","Moderate","Active","Very Active"], index=2)
    p["goal"] = st.selectbox("Goal", ["Weight Loss","Maintenance","Muscle Gain"], index=["Weight Loss","Maintenance","Muscle Gain"].index(p["goal"]))
    
    bmr = calculate_bmr(p["weight"], p["height"], p["age"], p["sex"])
    tdee = calculate_tdee(bmr, p["activity"])
    target = calculate_goals(tdee, p["weight"], p["goal"])
    st.session_state.daily_target = target
    
    st.divider()
    st.metric("TDEE", f"{tdee:.0f} kcal")
    st.json(target)
    
    if st.button("🗑️ Clear Today's Log"):
        st.session_state.meals = [m for m in st.session_state.meals if m["date"]!=date.today().isoformat()]
        st.rerun()

# --------- MAIN HEADER ---------
st.title("🥗 Live AI Nutrition & Smart Shopping Assistant")
st.caption("Live camera • Nutrition5k • OCR Label Scanner • AI Coach • Shopping - Built with Streamlit")

# Load models (cached, lazy)
nutrition_model_data = load_nutrition_model()
food_classifier = load_food_classifier()
ocr_reader = load_ocr_reader()

totals = get_totals()
remaining = {k: target[k]-totals.get(k,0) for k in ["calories","protein","carb","fat"]}

# Top KPI Row
c1,c2,c3,c4 = st.columns(4)
c1.metric("Calories Left", f"{remaining['calories']:.0f} kcal", f"{totals['calories']:.0f} eaten")
c2.metric("Protein Left", f"{remaining['protein']:.0f} g", f"{totals['protein']:.0f}g")
c3.metric("Carbs Left", f"{remaining['carb']:.0f} g", f"{totals['carb']:.0f}g")
c4.metric("Fat Left", f"{remaining['fat']:.0f} g", f"{totals['fat']:.0f}g")

tabs = st.tabs(["📷 Live Scanner", "🏷️ Label Scanner", "📊 Dashboard", "🤖 AI Coach", "🛒 Smart Shopping"])

# ===== TAB 1 Live Scanner =====
with tabs[0]:
    colA, colB = st.columns([1,1])
    with colA:
        st.subheader("Live Camera Food Recognition")
        st.write("This replicates your 'Let's Check Your Meal Together' UI")
        cam = st.camera_input("Take a picture of your meal")
        upl = st.file_uploader("Or upload food image", type=["jpg","png","jpeg"], key="food_up")
        img_file = cam if cam else upl
        portion = st.slider("Portion size (grams)", 50, 800, 350, step=10)
    with colB:
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            st.image(pil_img, caption="Captured Meal", use_column_width=True)
            
            with st.spinner("Analyzing with Vision models..."):
                # Try RGBD model first
                est = predict_with_rgbd_model(pil_img, nutrition_model_data)
                clf_label = "salad"
                clf_conf = 0.0
                if food_classifier:
                    try:
                        res = food_classifier(pil_img)
                        clf_label = res[0]["label"]
                        clf_conf = res[0]["score"]
                        st.info(f"Classifier: **{clf_label}** ({clf_conf:.1%})")
                    except Exception as e:
                        st.warning(f"Classifier error: {e}")
                
                if est is None:
                    est = estimate_from_food_db(clf_label, portion)
                else:
                    # scale RGBD result to portion if user changed
                    scale = portion / max(est["mass"],1)
                    for k in ["calories","protein","fat","carb"]:
                        est[k]=round(est[k]*scale,1)
                    est["mass"]=portion
                    est["name"]=f"{clf_label.title()} ({est['name']})"
                
                st.markdown(f"""
                <div class="food-card">
                    <h3>{est['name']} - {est['mass']}g</h3>
                    <span class="macro-badge" style="background:#e8f5e9">🌾 {est['carb']}% Carbs: {est['carb']}g</span>
                    <span class="macro-badge" style="background:#fff3e0">💧 {est['fat']}g Fats</span>
                    <span class="macro-badge" style="background:#ffebee">🍬 Sugar: {est.get('sugar',0)}g</span>
                    <h2 style="margin-top:15px">{est['calories']} kcal | P:{est['protein']}g C:{est['carb']}g F:{est['fat']}g</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Ingredient flower chart mock (as in screenshot 3)
                fig = go.Figure(data=[go.Pie(labels=["Rice","Salmon","Cucumber","Spinach","Lettuce","Sesame"], values=[42.9,28.6,14.3,4.4,4.2,5.7], hole=0.6)])
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("➕ Add to Today's Log", type="primary", use_container_width=True):
                    st.session_state.meals.append({
                        "id": len(st.session_state.meals),
                        "date": date.today().isoformat(),
                        "time": datetime.now().strftime("%H:%M"),
                        "name": est["name"],
                        "nutrition": est,
                    })
                    st.success(f"Added {est['name']}!")
                    st.balloons()
        else:
            st.info("📸 Start camera to scan your meal. Model: "+("Nutrition5k ConvNeXt FOUND" if nutrition_model_data[0] else "FOOD-DB Demo Mode"))

# ===== TAB 2 Label Scanner =====
with tabs[1]:
    st.subheader("Nutrition Facts Label Detection + OCR")
    col1,col2 = st.columns(2)
    with col1:
        cam2 = st.camera_input("Photograph Nutrition Facts table", key="label_cam")
        upl2 = st.file_uploader("Upload label image", type=["jpg","png"], key="label_up")
        img2_file = cam2 if cam2 else upl2
    with col2:
        if img2_file:
            pil2 = Image.open(img2_file).convert("RGB")
            st.image(pil2, caption="Label Image", use_column_width=True)
            if st.button("🔍 Extract with OCR"):
                with st.spinner("Running OCR..."):
                    if ocr_reader:
                        try:
                            arr = np.array(pil2)
                            texts = ocr_reader.readtext(arr, detail=0)
                            full_text = "\n".join(texts)
                            st.code(full_text)
                            parsed = parse_nutrition_label_text(full_text)
                        except Exception as e:
                            st.error(f"OCR failed {e}")
                            parsed = parse_nutrition_label_text("")
                    else:
                        st.warning("easyocr not installed - using regex demo on blank. On Cloud it will work after pip install.")
                        parsed = {
                            "calories":250,"total_fat_g":10,"saturated_fat_g":3,"protein_g":8,
                            "total_carbohydrates_g":30,"total_sugars_g":8,"sodium_mg":300,"raw_text":"demo"
                        }
                    st.session_state["last_label"] = parsed
                    st.json(parsed)
                    
                    # convert to meal
                    if parsed.get("calories"):
                        nutrition = {
                            "calories": parsed.get("calories",0),
                            "fat": parsed.get("total_fat_g",0),
                            "protein": parsed.get("protein_g",0),
                            "carb": parsed.get("total_carbohydrates_g",0),
                            "fiber": parsed.get("dietary_fiber_g",0) or 0,
                            "sugar": parsed.get("total_sugars_g",0) or 0,
                            "mass": 100,
                            "name": "Packaged Product (OCR)"
                        }
                        if st.button("Add Label as Meal"):
                            st.session_state.meals.append({
                                "id": len(st.session_state.meals),
                                "date": date.today().isoformat(),
                                "time": datetime.now().strftime("%H:%M"),
                                "name": nutrition["name"],
                                "nutrition": nutrition
                            })
                            st.success("Added!")
        else:
            st.info("Take a photo of the Nutrition Facts table. The system uses OCR + Table Recognition to extract calories, fat, carbs, sugar...")

# ===== TAB 3 Dashboard =====
with tabs[2]:
    st.subheader("Live Web Dashboard - Daily Progress")
    left,right = st.columns([1,2])
    with left:
        totals = get_totals()
        # Donut remaining
        fig = go.Figure(data=[go.Pie(
            labels=["Eaten","Remaining"],
            values=[max(totals["calories"],0), max(target["calories"]-totals["calories"],0)],
            hole=0.7, marker_colors=["#66bb6a","#e0e0e0"]
        )])
        fig.update_layout(title=f"Calories {totals['calories']:.0f}/{target['calories']}", height=250, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
        
        for macro in ["protein","carb","fat"]:
            pct = min(totals[macro]/target[macro],1.0) if target[macro]>0 else 0
            st.write(f"**{macro.title()}** {totals[macro]:.0f}/{target[macro]}g")
            st.progress(pct)
    
    with right:
        meals_today = get_today_meals()
        if meals_today:
            df = pd.DataFrame([{"Time":m["time"],"Food":m["name"],"kcal":m["nutrition"]["calories"],"P":m["nutrition"]["protein"],"C":m["nutrition"]["carb"],"F":m["nutrition"]["fat"]} for m in meals_today])
            st.dataframe(df, use_container_width=True)
            fig2 = px.bar(df, x="Food", y=["P","C","F"], barmode="group", title="Macros by Meal")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No meals logged yet. Go to Live Scanner!")
    
    st.divider()
    st.subheader("Meal Suggestions based on Remaining Macros")
    rem = remaining
    suggestions = []
    if rem["protein"]>20: suggestions.append("🔥 High Protein: Grilled Chicken 200g + Rice 100g (380 kcal, 42g P)")
    if rem["carb"]<50: suggestions.append("🥑 Low Carb: Avocado + Eggs (250 kcal, 15g C)")
    if rem["calories"]>400: suggestions.append("🍝 Balanced: Spicy Tomato Fusilli 250g (395 kcal)")
    if rem["calories"]<150: suggestions.append("🍎 Light: Apple + Greek Yogurt (120 kcal)")
    for s in suggestions: st.write(s)

# ===== TAB 4 AI Coach =====
with tabs[3]:
    st.subheader("AI Nutrition Chatbot - Talk to your data")
    st.write("Router intents: daily_summary, remaining_macros, portion_calculation, food_comparison...")
    
    # Voice Interaction
    st.markdown("#### 🎤 Voice Interaction")
    audio = st.audio_input("Ask with your voice")
    if audio:
        st.audio(audio)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio) as src:
                aud = r.record(src)
                text = r.recognize_google(aud)
                st.success(f"Transcribed: {text}")
                st.session_state.chat_history.append({"role":"user","content":text})
                ans = generate_coach_response(text)
                st.session_state.chat_history.append({"role":"assistant","content":ans})
        except Exception as e:
            st.warning(f"Voice transcription needs SpeechRecognition + internet: {e}. You can type instead.")
    
    # Chat UI
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    prompt = st.chat_input("e.g., How many calories do I have left? What should I eat for dinner?")
    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        ans = generate_coach_response(prompt)
        st.session_state.chat_history.append({"role":"assistant","content":ans})
        with st.chat_message("assistant"): st.markdown(ans)

# ===== TAB 5 Shopping =====
with tabs[4]:
    st.subheader("AI Smart Shopping - Compare & Better Alternatives")
    cA,cB = st.columns(2)
    with cA:
        st.markdown("**Product Comparison**")
        prod1 = st.selectbox("Product A", list(FOOD_DB.keys()), index=0)
        prod2 = st.selectbox("Product B", list(FOOD_DB.keys()), index=1)
        if st.button("Compare"):
            a=FOOD_DB[prod1]; b=FOOD_DB[prod2]
            comp_df = pd.DataFrame([a,b], index=[prod1,prod2])
            st.dataframe(comp_df)
            score_a = a["protein"]*2 - a["sugar"] - a["fat"]
            score_b = b["protein"]*2 - b["sugar"] - b["fat"]
            if score_a>score_b:
                st.success(f"✅ **{prod1}** is healthier (score {score_a:.1f} vs {score_b:.1f})")
                st.write(f"Swap tip: Choose {prod1} over {prod2} for more protein & less sugar")
            else:
                st.success(f"✅ **{prod2}** is healthier")
    with cB:
        st.markdown("**Shopping List**")
        new_item = st.text_input("Add food")
        if st.button("Add to List") and new_item:
            st.session_state.shopping_list.append(new_item)
        for i, item in enumerate(st.session_state.shopping_list):
            colx, coly = st.columns([4,1])
            colx.checkbox(item, key=f"shop_{i}")
            if coly.button("❌", key=f"del_{i}"):
                st.session_state.shopping_list.pop(i)
                st.rerun()
        st.write("Better alternatives AI:")
        st.info("Cake 🍰 → Greek Yogurt + Berries (save 200 kcal, +10g protein)\n\nSoda → Sparkling Water + Lemon\n\nWhite Rice → Quinoa (more fiber & protein)")

st.divider()
st.caption("Built with Nutrition5k + GLM-OCR + Qwen Chatbot logic | For demo, model weights optional | © Live AI Nutrition")
