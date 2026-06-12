import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from src.model import CIFAKEDetector
from src.dataset import build_predict_transform


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model
# -----------------------------
model = CIFAKEDetector(
    num_classes=2,
    pretrained=False
).to(device)

checkpoint_path = "outputs/best_model.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

model.eval()

transform = build_predict_transform()


# -----------------------------
# Prediction Function
# -----------------------------
def predict(image: Image.Image):

    if image is None:
        return "No image uploaded"

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    fake_prob = float(probs[0][0])
    real_prob = float(probs[0][1])

    return {
        "AI Generated": fake_prob,
        "Real": real_prob
    }


# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI vs Real Image Detector",
    description="Upload image to detect whether it is AI generated or real."
)

demo.launch()
