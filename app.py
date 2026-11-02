# Mask check (Streamlit). Flow: load the CNN and the OpenCV face detector once -> user uploads or captures a photo -> find faces (or use the whole photo) -> crop each face, resize to 128x128 and scale pixels to 0-1 exactly like training -> CNN returns P(no mask) -> compare with the threshold -> draw labelled boxes and show a verdict card per face plus a summary bar.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # keep TensorFlow startup logs quiet
import cv2, numpy as np, streamlit as st, tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = 128
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_mask_model.keras")
OK_RGB, BAD_RGB = (14, 159, 142), (229, 72, 77)  # box colours drawn on the photo

st.set_page_config(page_title="Mask check", page_icon="😷", layout="wide")

# ---------- Styling: fonts, colours, hero, result cards ----------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');
:root{--ink:#0F2A33;--mist:#EEF3F4;--line:#D5E0E3;--ok:#0E9F8E;--bad:#E5484D;--glow:#7DE3D1;--muted:#5B7279}
.stApp{background:var(--mist)}
.stApp,.stApp p,.stApp label,.stApp button{font-family:'DM Sans',system-ui,sans-serif}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{position:relative;background:var(--ink);color:#EAF4F3;border-radius:24px;padding:2.6rem 2.8rem 2.3rem;margin-bottom:1.8rem}
.hero:before,.hero:after{content:"";position:absolute;width:38px;height:38px;border:3px solid var(--glow)}
.hero:before{top:18px;left:18px;border-right:0;border-bottom:0;border-top-left-radius:10px}
.hero:after{bottom:18px;right:18px;border-left:0;border-top:0;border-bottom-right-radius:10px}
.hero .title{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-weight:800;font-size:clamp(2.4rem,5vw,3.8rem);line-height:1;letter-spacing:-.02em;color:#fff;margin:0 0 .9rem}
.hero p{max-width:34rem;font-size:1.08rem;line-height:1.55;color:#B9D3D6;margin:0 0 1.3rem}
.chips span{display:inline-block;border:1px solid #2E5560;color:#CFE6E8;border-radius:999px;padding:.28rem .85rem;margin:0 .5rem .4rem 0;font-size:.85rem}
.panel-title{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-weight:800;font-size:1.35rem;color:var(--ink);margin:0 0 .7rem}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:2px dashed #9BB8BE;border-radius:18px}
[data-testid="stImage"] img{border-radius:18px;border:1px solid var(--line)}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:16px}
.summary{font-size:1.08rem;color:var(--ink);margin:0 0 .6rem}
.split{display:flex;height:12px;border-radius:99px;overflow:hidden;background:#DCE6E8;margin-bottom:1rem}
.split i{display:block;height:100%}
.split .a{background:var(--ok)}
.split .b{background:var(--bad)}
.face{display:flex;align-items:center;gap:1rem;background:#fff;border:1px solid var(--line);border-radius:18px;padding:.9rem 1.1rem;margin-top:.7rem}
.face.ok{--c:var(--ok)}
.face.bad{--c:var(--bad)}
.ring{width:64px;height:64px;flex:none;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) var(--p),#DCE6E8 0)}
.ring b{width:48px;height:48px;border-radius:50%;background:#fff;display:grid;place-items:center;font-size:.9rem;color:var(--ink)}
.verdict{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-weight:800;font-size:1.4rem;line-height:1.1;color:var(--c)}
.sub{color:var(--muted);font-size:.92rem;margin-top:.2rem}
.empty{border:2px dashed #9BB8BE;border-radius:22px;padding:3.4rem 1.6rem;text-align:center;color:var(--muted);background:rgba(255,255,255,.55)}
.empty b{display:block;font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.35rem;color:var(--ink);margin-bottom:.4rem}
.note{color:var(--muted);font-size:.85rem;margin-top:2rem}
"""
html = lambda s: st.markdown(s, unsafe_allow_html=True)
html("<style>" + CSS + "</style>")

# ---------- Model and face detector (loaded once, then cached) ----------
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    classifier = getattr(cv2, "CascadeClassifier", None)
    data = getattr(cv2, "data", None)
    haarcascades = getattr(data, "haarcascades", None)
    if not callable(classifier):
        return model, None, "OpenCV face detection is unavailable. Reinstall opencv-python-headless and reboot the app."
    if not haarcascades:
        return model, None, "OpenCV's Haar cascade files are unavailable. Reinstall opencv-python-headless and reboot the app."

    cascade_path = os.path.join(haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.isfile(cascade_path):
        return model, None, "The OpenCV face-detection file is missing. Reinstall opencv-python-headless and reboot the app."

    cascade = classifier(cascade_path)
    if cascade.empty():
        return model, None, "OpenCV could not load its face detector. Reinstall opencv-python-headless and reboot the app."
    return model, cascade, None

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Place face_mask_model.keras in the same folder as app.py.")
    st.stop()
model, cascade, detector_warning = load_assets()
if detector_warning:
    st.warning(detector_warning + " The app will classify the whole image instead.")

# ---------- Image helpers ----------
def read_image(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")  # fix phone-photo rotation, force RGB
    img.thumbnail((1280, 1280))  # cap the size so detection stays fast
    return np.array(img)

def analyse(rgb, find_faces, threshold):
    boxes = []
    if find_faces and cascade is not None:
        found = cascade.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(60, 60))
        boxes = sorted(tuple(int(v) for v in b) for b in found)  # left to right
    whole = not boxes
    if whole: boxes = [(0, 0, rgb.shape[1], rgb.shape[0])]
    batch = np.stack([cv2.resize(rgb[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE)) for x, y, w, h in boxes]).astype("float32") / 255.0
    probs = model.predict(batch, verbose=0).ravel()  # P(no mask) for every face in one call
    results = [{"box": b, "masked": bool(p < threshold), "conf": float(1 - p if p < threshold else p)} for b, p in zip(boxes, probs)]
    return results, whole

def annotate(rgb, results):
    out, scale = rgb.copy(), max(0.5, rgb.shape[1] / 1100)
    for i, r in enumerate(results, 1):
        x, y, w, h = r["box"]; col = OK_RGB if r["masked"] else BAD_RGB
        text = f"{i}  {'Mask' if r['masked'] else 'No mask'} {r['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        top = max(y - th - 12, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, max(2, rgb.shape[1] // 300))
        cv2.rectangle(out, (x, top), (x + tw + 12, top + th + 12), col, -1)
        cv2.putText(out, text, (x + 6, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return out

def card(i, n, r):
    cls, text = ("ok", "Mask on") if r["masked"] else ("bad", "No mask")
    return (f'<div class="face {cls}"><div class="ring" style="--p:{r["conf"]*100:.0f}%"><b>{r["conf"]:.0%}</b></div>'
            f'<div><div class="verdict">{text}</div><div class="sub">Face {i} of {n}</div></div></div>')

# ---------- Page layout ----------
html('<div class="hero"><div class="title" role="heading" aria-level="1">Mask check</div>'
     '<p>Add a photo and the model finds every face, then tells you whether a mask is on.</p>'
     '<div class="chips"><span>Custom CNN</span><span>Haar face detection</span><span>TensorFlow</span></div></div>')

left, right = st.columns([5, 6], gap="large")

with left:
    html('<div class="panel-title">Add a photo</div>')
    source = st.radio("Source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    file = (st.file_uploader("Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed") if source == "Upload"
            else st.camera_input("Camera", label_visibility="collapsed"))
    with st.expander("Detection settings"):
        find_faces = st.toggle("Find faces first", True, help="Turn off for close-up face photos, like the training images.")
        threshold = st.slider("No-mask threshold", 0.1, 0.9, 0.5, 0.05, help="The model outputs P(no mask). Above this value a face is labelled No mask.")
    with st.expander("How this works"):
        st.markdown("1. **Find faces.** OpenCV locates each face in the photo.\n"
                    "2. **Prepare each face.** It is cropped, resized to 128×128 and scaled to 0–1, exactly like the training data.\n"
                    "3. **Classify.** The CNN returns the probability of no mask, and the threshold turns it into a verdict.")

with right:
    html('<div class="panel-title">Result</div>')
    if not file:
        html('<div class="empty"><b>Your result shows up here</b>Add a photo on the left. Clear, front-facing, well-lit photos work best.</div>')
    else:
        rgb = read_image(file)
        with st.spinner("Checking faces..."):
            results, whole = analyse(rgb, find_faces, threshold)
        n, m = len(results), sum(r["masked"] for r in results)
        html(f'<div class="summary">{n} face{"s" if n != 1 else ""} checked: {m} with a mask, {n - m} without.</div>'
             f'<div class="split"><i class="a" style="width:{100*m/n:.1f}%"></i><i class="b" style="width:{100*(n-m)/n:.1f}%"></i></div>')
        st.image(annotate(rgb, results))
        if whole and find_faces and cascade is not None:
            st.warning("No face was found, so the whole photo was checked. Try a clearer, front-facing photo.")
        html("".join(card(i, n, r) for i, r in enumerate(results, 1)))

html('<div class="note">A learning project. Results are estimates and should not be used for safety decisions.</div>')
