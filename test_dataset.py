import numpy as np
import torch
import os
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from src.dataset import OIDataset
from src.process import evaluate
from src.transform import SimpleTransformer
from src.utils import generate_dboxes, Encoder
from src.model import SSD
from src.dataset import collate_fn

classes = ['Background', 'Panda', 'Scissors', 'Snake']
input_folder = 'src/oiddata/test/Scissors/'
cls_threshold = 0.3
nms_threshold = 0.5  # Non-Maximum Suppression
model_path = 'trained_models/SSD.pth'
output_path = 'predictions/Scissors/'


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


def draw_prediction_one(path):
    img = Image.open(path).convert("RGB")
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
        output_img = cv2.imread(path)
        if len(loc) > 0:
            height, width, _ = output_img.shape
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = classes[lb]
                color = (255, 0, 0)
                xmin, ymin, xmax, ymax = box
                xmin, ymin, xmax, ymax = ensure_legal(xmin, ymin, xmax, ymax, width, height)
                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        output = output_path + "{}_prediction.jpg".format(os.path.splitext(os.path.split(path)[1])[0])
        cv2.imwrite(output, output_img)


def evaluate_test_dataset():
    test_params = {"batch_size": 4,
                   "shuffle": True,
                   "drop_last": False,
                   "num_workers": 4,
                   "collate_fn": collate_fn}
    test_set = OIDataset(transformer, test=True)
    test_loader = DataLoader(test_set, **test_params)
    evaluate(model, test_loader, encoder, 0.45)


if __name__ == "__main__":
    model = SSD()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    transformer = SimpleTransformer(dboxes, eval=True)
    encoder = Encoder(dboxes)

    # evaluate_test_dataset()

    for image in tqdm(os.listdir(input_folder)):
        draw_prediction_one(input_folder + image)
