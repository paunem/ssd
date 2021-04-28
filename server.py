from flask import Flask, request
import numpy as np
import torch
from PIL import Image
import json

from src.transform import SimpleTransformer
from src.utils import generate_dboxes, Encoder
from src.model import SSD

app = Flask(__name__)

classes = ['Background', 'Panda', 'Scissors', 'Snake']
model_path = 'trained_models/SSD.pth'
cls_threshold = 0.3
nms_threshold = 0.5  # Non-Maximum Suppression

model = SSD()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
    model.cuda()
model.eval()
dboxes = generate_dboxes()
transformer = SimpleTransformer(dboxes, eval=True)
encoder = Encoder(dboxes)


def ensure_legal(xmin, ymin, xmax, ymax, width, height):
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > width:
        xmax = width
    if ymax > height:
        ymax = height
    return xmin, ymin, xmax, ymax


@app.route('/predict', methods=['POST'])
def rest():
    r = request
    response = []

    img = Image.open(r.files['snake'].stream).convert("RGB")
    width, height = img.size
    img, _, _ = transformer(img, torch.zeros(1, 4), torch.zeros(1))

    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        ploc, plabel = model(img.unsqueeze(dim=0))
        result = encoder.decode_batch(ploc, plabel, nms_threshold, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]

        if len(loc) > 0:
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = classes[lb]
                xmin, ymin, xmax, ymax = box
                xmin, ymin, xmax, ymax = ensure_legal(xmin, ymin, xmax, ymax, width, height)

                response.append({'label': category, 'prob': float(pr), 'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]})

    return json.dumps(response)
