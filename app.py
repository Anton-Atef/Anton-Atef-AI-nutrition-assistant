import os, io, re, json, math, random
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import cv2

import torch
import torch.nn as nn
import timm

from huggingface_hub import hf_hub_download, InferenceClient


# =========================================================
# 🔧 CONFIG (HF)
# =========================================================
MODEL_REPO_ID   = "Anton-Atef/AI-nutrition-assistant"
MODEL_FILENAME  = "best_nutrition_rgbd.pt"

CHAT_MODEL      = "Qwen/Qwen2.5-7B-Instruct"
OCR_VISION_MODEL = "microsoft/trocr-large-printed"   # supported by HF serverless
ASR_MODEL       = "openai/whisper-large-v3"

CFG_IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_hf_token() -> Optional[str]:
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None
    return token or os.environ.get("HF_TOKEN")


def make_hf_client(model_id: str):
    token = get_hf_token()
    if not token:
        return None
    try:
        # New hub versions
        return InferenceClient(model=model_id, token=token, provider="hf-inference")
    except TypeError:
        # Older hub versions
        return InferenceClient(model=model_id, token=token)


@st.cache_resource
def hf_client_chat():
    return make_hf_client(CHAT_MODEL)


@st.cache_resource
def hf_client_ocr():
    return make_hf_client(OCR_VISION_MODEL)


@st.cache_resource
def hf_client_asr():
    return make_hf_client(ASR_MODEL)


# =========================================================
# --------- PAGE CONFIG ---------
# =========================================================
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

.header-wrap{
    padding: 18px 18px;
    border-radius: 20px;
    background: linear-gradient(-45deg, #00C853, #76FF03, #1B5E20, #00E676);
    background-size: 400% 400%;
    animation: grad 10s ease infinite;
    color:white;
    box-shadow: 0 14px 34px rgba(0,0,0,0.18);
    margin-bottom: 16px;
}
@keyframes grad {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.food-card {
    background: rgba(255,255,255,0.92);
    border-radius: 20px; padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom:15px;
    border: 1px solid rgba(0,200,83,0.15);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.food-card:hover{
    transform: translateY(-4px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.10);
}

.macro-badge {
    display:inline-block; padding:8px 14px; border-radius:12px;
    font-weight:600; font-size:14px; margin:5px;
    border: 1px solid rgba(0,0,0,0.06);
}

.pulse-dot {
    height:10px; width:10px; background:#ff1744; border-radius:50%;
    display:inline-block; margin-right:8px;
    animation: pulse 1.3s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,23,68,0.55); }
    70% { box-shadow: 0 0 0 12px rgba(255,23,68,0.00); }
    100% { box-shadow: 0 0 0 0 rgba(255,23,68,0.00); }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# --------- CONSTANTS / DEMO FOOD DB ---------
# =========================================================
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


# =========================================================
# --------- SESSION STATE ---------
# =========================================================
defaults = {
    "meals": [],
    "shopping_list": [],
    "chat_history": [],
    "user_profile": {"age": 25, "sex": "Male", "weight": 70.0, "height": 175.0, "activity": "Moderate", "goal": "Maintenance"},
    "daily_target": None,
    "ocr_history": [],
    "last_label": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# --------- HELPERS - NUTRITION MATH ---------
# =========================================================
def calculate_bmr(w, h, age, sex):
    # (your original formula)
    if sex == "Male":
        return 88.362 + (13.397 * w) + (4.799 * h) - (5.677 * age)
    return 447.593 + (9.247 * w) + (3.098 * h) - (4.330 * age)


def calculate_tdee(bmr, activity):
    mult = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
    return bmr * mult.get(activity, 1.55)


def calculate_goals(tdee, weight, goal):
    if goal == "Weight Loss":
        cal = tdee - 500
    elif goal == "Muscle Gain":
        cal = tdee + 300
    else:
        cal = tdee

    if goal == "Muscle Gain":
        p = weight * 2.2
    elif goal == "Weight Loss":
        p = weight * 2.0
    else:
        p = weight * 1.8

    f = (cal * 0.25) / 9
    c = (cal - (p * 4 + f * 9)) / 4
    return {"calories": round(cal), "protein": round(p), "fat": round(f), "carb": round(c), "fiber": 25}


def get_today_meals():
    today = date.today().isoformat()
    return [m for m in st.session_state.meals if m.get("date") == today]


def get_totals():
    meals = get_today_meals()
    tot = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carb": 0.0, "fiber": 0.0}
    for m in meals:
        n = m["nutrition"]
        for k in tot:
            tot[k] += float(n.get(k, 0) or 0)
    return tot


# =========================================================
# --------- RGB-D MODEL (your checkpoint) ---------
# =========================================================
class NutritionNet(nn.Module):
    def __init__(self, backbone_name="convnext_small", in_chans=4, out_dim=5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


@st.cache_resource
def load_nutrition_model():
    token = get_hf_token()
    try:
        ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            token=token
        )
    except Exception as e:
        st.warning(f"Could not download Nutrition5k checkpoint: {e}")
        return None, None, None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
        y_std = np.array(ckpt["y_std"], dtype=np.float32)
        cols = ckpt.get("target_cols", ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"])
        model = NutritionNet(
            ckpt.get("backbone", "convnext_small"),
            in_chans=ckpt.get("in_chans", 4),
            out_dim=len(cols)
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, (y_mean, y_std), cols
    except Exception as e:
        st.warning(f"Could not load Nutrition5k checkpoint: {e}")
        return None, None, None


@st.cache_resource
def load_food_classifier():
    # Optional: transformer classifier (if transformers installed)
    try:
        from transformers import pipeline
        pipe = pipeline("image-classification", model="nateraw/food", top_k=3)
        return pipe
    except Exception:
        return None


def predict_with_rgbd_model(pil_img: Image.Image, model_data):
    if model_data[0] is None:
        return None

    model, (y_mean, y_std), cols = model_data
    try:
        # pad-resize similar to your training style (simple center pad)
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = CFG_IMG_SIZE / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((CFG_IMG_SIZE, CFG_IMG_SIZE, 3), dtype=np.uint8)
        top, left = (CFG_IMG_SIZE - nh) // 2, (CFG_IMG_SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = rgb_r

        rgb_f = canvas.astype(np.float32) / 255.0
        rgb_f = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD

        # IMPORTANT: neutral depth should be 0.5 so (depth-0.5)/0.25 == 0
        depth = np.full((CFG_IMG_SIZE, CFG_IMG_SIZE, 1), 0.5, dtype=np.float32)
        depth = (depth - 0.5) / 0.25

        img4 = np.concatenate([rgb_f, depth], axis=2).astype(np.float32)
        x = torch.from_numpy(img4).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred_std = model(x).cpu().numpy()[0]

        pred = pred_std * y_std + y_mean
        mapping = {c: float(v) for c, v in zip(cols, pred)}

        return {
            "name": "AI Estimated Dish",
            "mass": float(mapping.get("total_mass", 250)),
            "calories": float(mapping.get("total_calories", 200)),
            "fat": float(mapping.get("total_fat", 10)),
            "carb": float(mapping.get("total_carb", 18)),
            "protein": float(mapping.get("total_protein", 12)),
            "fiber": 2.0,
            "sugar": 3.0
        }
    except Exception as e:
        st.error(f"RGBD inference failed: {type(e).__name__}: {e}")
        return None


def estimate_from_food_db(label: str, grams: float):
    label = (label or "").lower()
    best = None
    for k, v in FOOD_DB.items():
        if k in label:
            best = v
            break
    if not best:
        best = FOOD_DB["salad"]

    factor = grams / 100.0
    return {
        "name": best["name"],
        "calories": round(best["calories"] * factor, 1),
        "protein": round(best["protein"] * factor, 1),
        "fat": round(best["fat"] * factor, 1),
        "carb": round(best["carb"] * factor, 1),
        "fiber": round(best.get("fiber", 0) * factor, 1),
        "sugar": round(best.get("sugar", 0) * factor, 1),
        "mass": float(grams)
    }


# =========================================================
# --------- OCR (AUTO-CROP + TrOCR + Qwen JSON) ---------
# =========================================================
EXPECTED_KEYS = [
    "product_name", "serving_size", "servings_per_container",
    "calories", "total_fat_g", "saturated_fat_g", "trans_fat_g",
    "cholesterol_mg", "sodium_mg", "total_carbohydrates_g",
    "dietary_fiber_g", "total_sugars_g", "added_sugars_g", "protein_g"
]

OCR_TO_JSON_PROMPT = """
Convert the OCR text of a Nutrition Facts label into a strict JSON object.

Use EXACTLY these keys:
- "product_name" (string or null)
- "serving_size" (string or null)
- "servings_per_container" (number only or null)
- "calories" (number only or null)
- "total_fat_g" (number only or null)
- "saturated_fat_g" (number only or null)
- "trans_fat_g" (number only or null)
- "cholesterol_mg" (number only or null)
- "sodium_mg" (number only or null)
- "total_carbohydrates_g" (number only or null)
- "dietary_fiber_g" (number only or null)
- "total_sugars_g" (number only or null)
- "added_sugars_g" (number only or null)
- "protein_g" (number only or null)

Rules:
1) Do NOT use % Daily Value numbers.
2) If not visible/reliable, set it to null. Never guess.
3) Return ONLY valid JSON (no markdown, no explanations).
""".strip()


def _extract_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _largest_contour_bbox(mask: np.ndarray, min_area: float):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area > best_area and area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            best = (x, y, w, h)
            best_area = area
    return best


def _detect_white_panel(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, W = bgr.shape[:2]
    # white: low sat, high value
    white_mask = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k, iterations=2)
    return _largest_contour_bbox(white_mask, min_area=0.06 * H * W)


def auto_crop_nutrition_facts(pil_img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Works well for images like your example:
    - red label on bottle + white Nutrition Facts panel
    Strategy:
      A) Find big RED region (label)
      B) Inside it, find WHITE rectangle (nutrition panel)
      C) Fallback: directly find biggest WHITE rectangle in whole image
    """
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    dbg = {"used_full_image": False, "found_label": False, "found_panel": False, "fallback_white_full": False}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # A) red mask
    mask1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=2)

    label_bbox = _largest_contour_bbox(red_mask, min_area=0.03 * H * W)

    if label_bbox is not None:
        dbg["found_label"] = True
        x, y, w, h = label_bbox
        pad = int(0.02 * max(H, W))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

        label = bgr[y0:y1, x0:x1].copy()
        lh, lw = label.shape[:2]

        # B) find white panel inside label
        panel_bbox = _detect_white_panel(label)
        if panel_bbox is not None:
            dbg["found_panel"] = True
            px, py, pw, ph = panel_bbox
            ppad = int(0.02 * max(lh, lw))
            px0 = max(0, px - ppad); py0 = max(0, py - ppad)
            px1 = min(lw, px + pw + ppad); py1 = min(lh, py + ph + ppad)
            panel = label[py0:py1, px0:px1].copy()
            out = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            return Image.fromarray(out), dbg

        # if panel not found, return label crop
        out = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out), dbg

    # C) fallback: biggest white panel in full image
    full_panel = _detect_white_panel(bgr)
    if full_panel is not None:
        dbg["fallback_white_full"] = True
        x, y, w, h = full_panel
        pad = int(0.02 * max(H, W))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        crop = bgr[y0:y1, x0:x1].copy()
        out = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out), dbg

    dbg["used_full_image"] = True
    return pil_img, dbg


def preprocess_label_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # upscale helps OCR a lot
    target_w = 1600
    if w < target_w:
        scale = target_w / max(1, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

    # mild sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return Image.fromarray(img)


def parse_nutrition_label_text_regex(text: str) -> Dict[str, Any]:
    # fallback parser (best-effort) if LLM parsing fails
    t = (text or "").lower()

    def find(pattern):
        m = re.search(pattern, t, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    return {
        "product_name": None,
        "serving_size": None,
        "servings_per_container": find(r"servings per container\D{0,15}(\d+\.?\d*)"),
        "calories": find(r"calories\D{0,10}(\d+\.?\d*)"),
        "total_fat_g": find(r"total fat\D{0,10}(\d+\.?\d*)\s*g"),
        "saturated_fat_g": find(r"saturated fat\D{0,10}(\d+\.?\d*)\s*g"),
        "trans_fat_g": find(r"trans fat\D{0,10}(\d+\.?\d*)\s*g"),
        "cholesterol_mg": find(r"cholesterol\D{0,10}(\d+\.?\d*)\s*mg"),
        "sodium_mg": find(r"sodium\D{0,10}(\d+\.?\d*)\s*mg"),
        "total_carbohydrates_g": find(r"(?:total\s*)?carbohydrate\D{0,10}(\d+\.?\d*)\s*g"),
        "dietary_fiber_g": find(r"(?:dietary\s*)?fiber\D{0,10}(\d+\.?\d*)\s*g"),
        "total_sugars_g": find(r"(?:total\s*)?sugars?\D{0,10}(\d+\.?\d*)\s*g"),
        "added_sugars_g": find(r"added sugars?\D{0,10}(\d+\.?\d*)\s*g"),
        "protein_g": find(r"protein\D{0,10}(\d+\.?\d*)\s*g"),
        "raw_text": (text or "")[:900],
    }


def _call_image_to_text(client: InferenceClient, img_bytes: bytes):
    # handle hub signature differences
    try:
        return client.image_to_text(image=img_bytes)
    except TypeError:
        return client.image_to_text(img_bytes)


def extract_nutrition_label_ai(pil_img: Image.Image) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Returns: (parsed_json, error_string, debug_dict)
    """
    dbg = {}
    token = get_hf_token()
    if not token:
        return None, "HF_TOKEN missing. Add it in Streamlit Secrets.", dbg

    ocr_client = hf_client_ocr()
    chat_client = hf_client_chat()
    if ocr_client is None or chat_client is None:
        return None, "HF clients not available. Check HF_TOKEN / huggingface_hub.", dbg

    try:
        # 1) crop
        crop_img, crop_dbg = auto_crop_nutrition_facts(pil_img)
        dbg["crop"] = crop_dbg

        # 2) preprocess
        prep_img = preprocess_label_for_ocr(crop_img)
        img_bytes = _pil_to_png_bytes(prep_img)

        # 3) OCR text (TrOCR)
        ocr_out = _call_image_to_text(ocr_client, img_bytes)

        if isinstance(ocr_out, list) and ocr_out:
            ocr_text = ocr_out[0].get("generated_text") or ocr_out[0].get("text") or str(ocr_out[0])
        elif isinstance(ocr_out, dict):
            ocr_text = ocr_out.get("generated_text") or ocr_out.get("text") or str(ocr_out)
        else:
            ocr_text = str(ocr_out)

        ocr_text = (ocr_text or "").strip()
        dbg["raw_ocr_text"] = ocr_text

        if len(ocr_text) < 10:
            return None, "OCR text too short. Take closer photo, reduce glare, keep label straight.", dbg

        # 4) Qwen parse OCR text -> JSON
        messages = [
            {"role": "system", "content": "You extract Nutrition Facts from OCR text. Output ONLY JSON."},
            {"role": "user", "content": OCR_TO_JSON_PROMPT + "\n\nOCR TEXT:\n" + ocr_text},
        ]
        resp = chat_client.chat_completion(messages=messages, max_tokens=520, temperature=0.1)
        llm_text = resp.choices[0].message["content"]
        dbg["raw_llm_text"] = llm_text

        parsed = _extract_json_object(llm_text)
        if not isinstance(parsed, dict):
            # fallback to regex
            parsed = parse_nutrition_label_text_regex(ocr_text)
            parsed["parse_fallback"] = "regex"
        else:
            parsed = {k: parsed.get(k, None) for k in EXPECTED_KEYS}
            parsed["parse_fallback"] = None

        return parsed, None, dbg

    except Exception as e:
        return None, f"OCR failed: {type(e).__name__}: {repr(e)}", dbg


# =========================================================
# --------- VOICE (Whisper via HF) ---------
# =========================================================
def transcribe_audio_hf(audio_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    client = hf_client_asr()
    if client is None:
        return None, "ASR requires HF_TOKEN in Streamlit Secrets."

    try:
        # best-effort across hub versions
        if hasattr(client, "automatic_speech_recognition"):
            res = client.automatic_speech_recognition(audio=audio_bytes)
            if isinstance(res, str):
                return res, None
            if isinstance(res, dict) and "text" in res:
                return res["text"], None

        if hasattr(client, "audio_to_text"):
            res = client.audio_to_text(audio_bytes)
            if isinstance(res, str):
                return res, None
            if isinstance(res, dict) and "text" in res:
                return res["text"], None

        return None, "ASR method not available in this huggingface_hub version."
    except Exception as e:
        return None, f"ASR failed: {type(e).__name__}: {repr(e)}"


# =========================================================
# --------- ROUTER (your code) ---------
# =========================================================
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
            "daily_summary": "get_daily_summary",
            "remaining_macros": "get_remaining_macros",
            "portion_calculation": "calculate_food_portion",
            "food_logging": "log_food",
            "recommendation": "recommend_meals",
            "food_comparison": "compare_foods",
            "user_targets": "get_user_targets",
            "food_history": "get_food_history",
        }

    def route(self, message: str) -> Route:
        text = re.sub(r"\s+", " ", message.lower().strip())
        scores = {intent: sum(bool(re.search(p, text)) for p in pats) for intent, pats in self.rules.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Route("general_nutrition_chat", None, 0.25)
        return Route(best, self.tool_map[best], min(0.95, 0.55 + scores[best] * 0.15))


router = AIRouter()


def hf_chat_reply(user_msg: str, ctx: Dict[str, Any]) -> Optional[str]:
    client = hf_client_chat()
    if client is None:
        return None
    try:
        messages = [
            {"role": "system", "content": "You are an AI Nutrition Coach. Use ONLY the provided context. Do not invent numbers."},
            {"role": "system", "content": "Trusted context JSON:\n" + json.dumps(ctx)},
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat_completion(messages=messages, max_tokens=340, temperature=0.6)
        return resp.choices[0].message["content"]
    except Exception:
        return None


def generate_coach_response(user_msg: str):
    totals = get_totals()
    target = st.session_state.daily_target or calculate_goals(2000, 70, "Maintenance")
    remaining = {k: float(target[k]) - totals.get(k, 0) for k in ["calories", "protein", "carb", "fat"]}
    route = router.route(user_msg)

    ctx = {
        "totals_today": totals,
        "target": target,
        "remaining": remaining,
        "meals_today": get_today_meals(),
        "goal": st.session_state.user_profile.get("goal"),
    }

    # deterministic answers first
    if route.intent == "daily_summary":
        return (
            f"**Today's Summary** 📊\n\n"
            f"You logged **{len(get_today_meals())}** meals.\n\n"
            f"- Calories: **{totals['calories']:.0f}/{target['calories']}** (remaining **{remaining['calories']:.0f}**)\n"
            f"- Protein: **{totals['protein']:.0f}g/{target['protein']}g** (remaining **{remaining['protein']:.0f}g**)\n"
            f"- Carbs: **{totals['carb']:.0f}g/{target['carb']}g**\n"
            f"- Fat: **{totals['fat']:.0f}g/{target['fat']}g**\n\n"
            f"You're doing well—want a suggestion to finish the day strong?"
        )

    if route.intent == "remaining_macros":
        return (
            f"Remaining today:\n\n"
            f"- 🔥 **{remaining['calories']:.0f} kcal**\n"
            f"- 💪 **{remaining['protein']:.0f}g protein**\n"
            f"- 🍞 **{remaining['carb']:.0f}g carbs**\n"
            f"- 🥑 **{remaining['fat']:.0f}g fat**\n\n"
            f"Tell me what food you’re craving and I’ll fit it into your macros."
        )

    if route.intent == "recommendation":
        if remaining["protein"] > 20:
            return (
                f"With **{remaining['calories']:.0f} kcal** left and **{remaining['protein']:.0f}g protein** needed, try:\n"
                f"- **Grilled chicken + rice + salad**\n"
                f"- **Greek yogurt + fruit**\n"
                f"- **Salmon bowl**"
            )
        return (
            f"With **{remaining['calories']:.0f} kcal** left, a lighter option works:\n"
            f"- **Salad + protein**\n"
            f"- **Apple + yogurt**"
        )

    if route.intent == "portion_calculation":
        return (
            f"Tell me the food name + package calories (or scan the label).\n"
            f"You have **{remaining['calories']:.0f} kcal** remaining."
        )

    if route.intent == "user_targets":
        return (
            f"Your daily targets ({st.session_state.user_profile['goal']}):\n"
            f"- **{target['calories']} kcal**\n"
            f"- **Protein {target['protein']}g**\n"
            f"- **Carbs {target['carb']}g**\n"
            f"- **Fat {target['fat']}g**"
        )

    # HF LLM fallback (if token exists)
    llm = hf_chat_reply(user_msg, ctx)
    if llm:
        return llm

    return (
        "I can help you track calories/macros, suggest meals, and scan nutrition labels.\n\n"
        "Try: “How many calories left today?” or “Recommend dinner based on my remaining macros.”"
    )


# =========================================================
# --------- SIDEBAR ---------
# =========================================================
with st.sidebar:
    st.title("👤 Your Profile")
    p = st.session_state.user_profile
    p["age"] = st.number_input("Age", 10, 90, int(p["age"]))
    p["sex"] = st.selectbox("Sex", ["Male", "Female"], index=0 if p["sex"] == "Male" else 1)
    p["weight"] = st.number_input("Weight (kg)", 30.0, 200.0, float(p["weight"]))
    p["height"] = st.number_input("Height (cm)", 100.0, 230.0, float(p["height"]))
    p["activity"] = st.selectbox("Activity", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], index=2)
    p["goal"] = st.selectbox("Goal", ["Weight Loss", "Maintenance", "Muscle Gain"],
                             index=["Weight Loss", "Maintenance", "Muscle Gain"].index(p["goal"]))

    bmr = calculate_bmr(p["weight"], p["height"], p["age"], p["sex"])
    tdee = calculate_tdee(bmr, p["activity"])
    target = calculate_goals(tdee, p["weight"], p["goal"])
    st.session_state.daily_target = target

    st.divider()
    st.metric("TDEE", f"{tdee:.0f} kcal")
    st.json(target)

    st.caption("🔐 Put HF_TOKEN in Streamlit Secrets to enable OCR/Chat/Voice + private model download.")
    if st.button("🗑️ Clear Today's Log", width="stretch"):
        st.session_state.meals = [m for m in st.session_state.meals if m["date"] != date.today().isoformat()]
        st.rerun()


# =========================================================
# --------- MAIN HEADER ---------
# =========================================================
st.markdown(
    """
<div class="header-wrap">
  <div style="font-size: 1.9rem; font-weight: 800;">🥗 Live AI Nutrition & Smart Shopping Assistant</div>
  <div style="opacity:0.95; margin-top:6px;">Live camera • Nutrition5k RGB-D • Label OCR • AI Coach • Shopping</div>
</div>
""",
    unsafe_allow_html=True
)

# Load models (cached)
nutrition_model_data = load_nutrition_model()
food_classifier = load_food_classifier()

totals = get_totals()
target = st.session_state.daily_target or {"calories": 2000, "protein": 130, "carb": 230, "fat": 65, "fiber": 25}
remaining = {k: float(target[k]) - totals.get(k, 0) for k in ["calories", "protein", "carb", "fat"]}

# KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Calories Left", f"{remaining['calories']:.0f} kcal", f"{totals['calories']:.0f} eaten")
c2.metric("Protein Left", f"{remaining['protein']:.0f} g", f"{totals['protein']:.0f}g")
c3.metric("Carbs Left", f"{remaining['carb']:.0f} g", f"{totals['carb']:.0f}g")
c4.metric("Fat Left", f"{remaining['fat']:.0f} g", f"{totals['fat']:.0f}g")


tabs = st.tabs(["📷 Live Scanner", "🏷️ Label Scanner (OCR)", "📊 Dashboard", "🤖 AI Coach + Voice", "🛒 Smart Shopping"])


# =========================================================
# ===== TAB 1 Live Scanner =====
# =========================================================
with tabs[0]:
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Live Camera Food Recognition")
        st.markdown('<span class="pulse-dot"></span> **Capture your meal**', unsafe_allow_html=True)
        cam = st.camera_input("Take a picture of your meal")
        upl = st.file_uploader("Or upload food image", type=["jpg", "png", "jpeg"], key="food_up")
        img_file = cam if cam else upl
        portion = st.slider("Portion size (grams)", 50, 800, 350, step=10)

    with colB:
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            st.image(pil_img, caption="Captured Meal", width="stretch")

            with st.spinner("Analyzing with Nutrition model + (optional) classifier..."):
                clf_label = "salad"
                clf_conf = 0.0
                if food_classifier is not None:
                    try:
                        res = food_classifier(pil_img)
                        clf_label = res[0]["label"]
                        clf_conf = float(res[0]["score"])
                        st.info(f"Classifier: **{clf_label}** ({clf_conf:.1%})")
                    except Exception as e:
                        st.warning(f"Classifier error: {e}")

                est = predict_with_rgbd_model(pil_img, nutrition_model_data)
                if est is None:
                    est = estimate_from_food_db(clf_label, portion)
                else:
                    # scale estimated dish to chosen portion
                    scale = float(portion) / max(float(est.get("mass", 250.0)), 1.0)
                    for k in ["calories", "protein", "fat", "carb", "fiber", "sugar"]:
                        est[k] = round(float(est.get(k, 0.0)) * scale, 1)
                    est["mass"] = float(portion)
                    est["name"] = f"{clf_label.title()} (AI Estimate)"

                st.markdown(
                    f"""
<div class="food-card">
  <h3 style="margin:0 0 8px 0;">{est['name']} — {est['mass']:.0f}g</h3>
  <span class="macro-badge" style="background:#e8f5e9;">🍞 Carbs: {est['carb']}g</span>
  <span class="macro-badge" style="background:#fff3e0;">🥑 Fat: {est['fat']}g</span>
  <span class="macro-badge" style="background:#ffebee;">💪 Protein: {est['protein']}g</span>
  <span class="macro-badge" style="background:#e3f2fd;">🌿 Fiber: {est.get('fiber',0)}g</span>
  <span class="macro-badge" style="background:#f3e5f5;">🍬 Sugar: {est.get('sugar',0)}g</span>
  <h2 style="margin-top:14px;">{est['calories']} kcal</h2>
</div>
""",
                    unsafe_allow_html=True,
                )

                # Ingredient donut (mock)
                fig = go.Figure(data=[
                    go.Pie(
                        labels=["Rice", "Protein", "Veggies", "Sauce", "Extras"],
                        values=[40, 30, 20, 5, 5],
                        hole=0.65
                    )
                ])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(fig, width="stretch")

                if st.button("➕ Add to Today's Log", type="primary", width="stretch"):
                    st.session_state.meals.append({
                        "id": len(st.session_state.meals),
                        "date": date.today().isoformat(),
                        "time": datetime.now().strftime("%H:%M"),
                        "name": est["name"],
                        "nutrition": est,
                    })
                    st.success(f"Added {est['name']}!")
                    try:
                        st.balloons()
                    except Exception:
                        pass
        else:
            st.info("📸 Start camera to scan your meal.")
            if nutrition_model_data[0] is None:
                st.warning("Nutrition5k checkpoint not loaded. App will use FOOD_DB fallback until model is available.")


# =========================================================
# ===== TAB 2 Label Scanner (OCR) =====
# =========================================================
with tabs[1]:
    st.subheader("Nutrition Facts Label — Auto Crop + OCR + JSON Extraction")
    st.caption("Best results: fill the frame with the Nutrition Facts panel, avoid glare, keep label straight.")

    col1, col2 = st.columns(2)
    with col1:
        cam2 = st.camera_input("Photograph Nutrition Facts table", key="label_cam")
        upl2 = st.file_uploader("Upload label image", type=["jpg", "png", "jpeg"], key="label_up")
        img2_file = cam2 if cam2 else upl2

    with col2:
        if img2_file:
            pil2 = Image.open(img2_file).convert("RGB")
            st.image(pil2, caption="Original Label Image", width="stretch")

            crop_preview, crop_dbg = auto_crop_nutrition_facts(pil2)
            st.image(
                crop_preview,
                caption=f"Auto-crop preview (label={crop_dbg.get('found_label')}, panel={crop_dbg.get('found_panel')}, fallback_white_full={crop_dbg.get('fallback_white_full')})",
                width="stretch"
            )

            if st.button("🔍 Extract with AI OCR", type="primary", width="stretch"):
                with st.spinner("Running auto-crop → OCR → JSON parsing..."):
                    parsed, err, dbg = extract_nutrition_label_ai(pil2)

                if err:
                    st.error(err)
                    with st.expander("Debug"):
                        st.write(dbg)
                else:
                    st.session_state.last_label = parsed
                    st.session_state.ocr_history.append({"time": datetime.now().isoformat(), "data": parsed, "debug": dbg})
                    st.success("Extracted ✅")

        else:
            st.info("Take a photo of the Nutrition Facts panel to extract calories, fat, carbs, sugar, etc.")

    if st.session_state.last_label:
        st.divider()
        st.markdown("### ✅ Extracted Nutrition JSON")
        st.code(json.dumps(st.session_state.last_label, indent=2), language="json")

        # Convert to meal (per serving; if user wants per 100g, you can add a scaling UI)
        parsed = st.session_state.last_label
        nutrition = {
            "name": (parsed.get("product_name") or "Packaged Product (OCR)"),
            "mass": 100.0,
            "calories": float(parsed.get("calories") or 0),
            "fat": float(parsed.get("total_fat_g") or 0),
            "protein": float(parsed.get("protein_g") or 0),
            "carb": float(parsed.get("total_carbohydrates_g") or 0),
            "fiber": float(parsed.get("dietary_fiber_g") or 0),
            "sugar": float(parsed.get("total_sugars_g") or 0),
        }

        name_override = st.text_input("Meal name", value=nutrition["name"])
        nutrition["name"] = name_override.strip() or nutrition["name"]

        if st.button("➕ Add Label as Meal", width="stretch"):
            st.session_state.meals.append({
                "id": len(st.session_state.meals),
                "date": date.today().isoformat(),
                "time": datetime.now().strftime("%H:%M"),
                "name": nutrition["name"],
                "nutrition": nutrition
            })
            st.success("Added ✅")


# =========================================================
# ===== TAB 3 Dashboard =====
# =========================================================
with tabs[2]:
    st.subheader("Live Web Dashboard - Daily Progress")
    left, right = st.columns([1, 2])

    with left:
        totals = get_totals()
        target = st.session_state.daily_target or target

        fig = go.Figure(data=[go.Pie(
            labels=["Eaten", "Remaining"],
            values=[
                max(float(totals["calories"]), 0.0),
                max(float(target["calories"]) - float(totals["calories"]), 0.0)
            ],
            hole=0.72, marker_colors=["#66bb6a", "#e0e0e0"]
        )])
        fig.update_layout(
            title=f"Calories {totals['calories']:.0f}/{target['calories']}",
            height=260, margin=dict(t=42, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, width="stretch")

        for macro in ["protein", "carb", "fat"]:
            pct = min(float(totals[macro]) / max(float(target[macro]), 1.0), 1.0)
            st.write(f"**{macro.title()}** {totals[macro]:.0f}/{target[macro]}g")
            st.progress(pct)

    with right:
        meals_today = get_today_meals()
        if meals_today:
            df = pd.DataFrame([
                {
                    "Time": m["time"],
                    "Food": m["name"],
                    "kcal": m["nutrition"]["calories"],
                    "P": m["nutrition"]["protein"],
                    "C": m["nutrition"]["carb"],
                    "F": m["nutrition"]["fat"]
                }
                for m in meals_today
            ])
            st.dataframe(df, width="stretch")
            fig2 = px.bar(df, x="Food", y=["P", "C", "F"], barmode="group", title="Macros by Meal")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No meals logged yet. Go to Live Scanner!")

    st.divider()
    st.subheader("Meal Suggestions based on Remaining Macros")
    totals = get_totals()
    rem = {
        "calories": float(target["calories"]) - totals["calories"],
        "protein": float(target["protein"]) - totals["protein"],
        "carb": float(target["carb"]) - totals["carb"],
        "fat": float(target["fat"]) - totals["fat"],
    }

    suggestions = []
    if rem["protein"] > 20:
        suggestions.append("🔥 High Protein: Grilled Chicken + Rice + Salad (fits protein)")
    if rem["carb"] < 50:
        suggestions.append("🥑 Lower Carb: Eggs + veggies + avocado")
    if rem["calories"] > 400:
        suggestions.append("🍝 Balanced: Pasta + lean protein + veg")
    if rem["calories"] < 150:
        suggestions.append("🍎 Light: Apple + Greek yogurt")

    for s in suggestions:
        st.write(s)


# =========================================================
# ===== TAB 4 AI Coach + Voice =====
# =========================================================
with tabs[3]:
    st.subheader("AI Nutrition Chatbot - Talk to your data")

    st.markdown("#### 🎙️ Voice Interaction (Whisper via HF)")
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Record your question")
        if audio is not None:
            if st.button("Transcribe Voice", width="stretch"):
                with st.spinner("Transcribing..."):
                    text, err = transcribe_audio_hf(audio.getvalue())
                if err:
                    st.error(err)
                else:
                    st.success(f"Transcribed: {text}")
                    st.session_state.chat_history.append({"role": "user", "content": text})
                    ans = generate_coach_response(text)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
    else:
        st.info("Your Streamlit version does not support audio_input.")

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g., How many calories do I have left? What should I eat for dinner?")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ans = generate_coach_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)


# =========================================================
# ===== TAB 5 Shopping =====
# =========================================================
with tabs[4]:
    st.subheader("AI Smart Shopping - Compare & Better Alternatives")
    cA, cB = st.columns(2)

    with cA:
        st.markdown("**Product Comparison**")
        prod1 = st.selectbox("Product A", list(FOOD_DB.keys()), index=0)
        prod2 = st.selectbox("Product B", list(FOOD_DB.keys()), index=1)

        if st.button("Compare", width="stretch"):
            a = FOOD_DB[prod1]
            b = FOOD_DB[prod2]
            comp_df = pd.DataFrame([a, b], index=[prod1, prod2])
            st.dataframe(comp_df, width="stretch")

            score_a = a["protein"] * 2 - a["sugar"] - a["fat"]
            score_b = b["protein"] * 2 - b["sugar"] - b["fat"]
            if score_a > score_b:
                st.success(f"✅ **{prod1}** is healthier (score {score_a:.1f} vs {score_b:.1f})")
                st.write(f"Swap tip: Choose **{prod1}** over **{prod2}** for more protein & less sugar.")
            else:
                st.success(f"✅ **{prod2}** is healthier (score {score_b:.1f} vs {score_a:.1f})")

    with cB:
        st.markdown("**Shopping List**")
        new_item = st.text_input("Add food")
        if st.button("Add to List", width="stretch") and new_item:
            st.session_state.shopping_list.append(new_item)

        for i, item in enumerate(list(st.session_state.shopping_list)):
            colx, coly = st.columns([4, 1])
            colx.checkbox(item, key=f"shop_{i}")
            if coly.button("❌", key=f"del_{i}", width="stretch"):
                st.session_state.shopping_list.pop(i)
                st.rerun()

        st.write("Better alternatives:")
        st.info(
            "Cake 🍰 → Greek Yogurt + Berries (save calories, +protein)\n\n"
            "Soda → Sparkling Water + Lemon\n\n"
            "White Rice → Quinoa (more fiber & protein)"
        )


st.divider()
st.caption("⚠️ General nutrition guidance only — not medical advice.")
