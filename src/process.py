import numpy as np
from tqdm.autonotebook import tqdm
import torch
import src.evaluation as e


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()
    for i, (img, _, gtlocation, gtlabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gtlocation = gtlocation.cuda()
            gtlabel = gtlabel.cuda()

        prlocation, prlabel = model(img)
        prlocation, prlabel = prlocation.float(), prlabel.float()
        gtlocation = gtlocation.transpose(1, 2).contiguous()
        loss = criterion(prlocation, prlabel, gtlocation, gtlabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, val_loader, encoder, nms_threshold):
    print('Calculating mAP')
    model.eval()
    detections = []
    gt = []
    height, width = 300, 300

    for nbatch, (img, img_id, bbox, label) in enumerate(val_loader):
        for i in range(len(img_id)):
            gt.append([img_id[i], round(float(bbox[i][0][0]) * height), round(float(bbox[i][0][1]) * width),
                       round(float(bbox[i][0][2]) * height), round(float(bbox[i][0][3]) * width), int(label[i][0])])

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            prlocation, prlabel = model(img)
            prlocation, prlabel = prlocation.float(), prlabel.float()

            for idx in range(prlocation.shape[0]):
                detections_loc = []
                ploc_i = prlocation[idx, :, :].unsqueeze(0)
                plabel_i = prlabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    # imgid, xmin, ymin, xmax, ymax, prob, label
                    detections_loc.append([img_id[idx], round(loc_[0] * 300), round(loc_[1] * 300),
                                           round(loc_[2] * 300), round(loc_[3] * 300), float(prob_), int(label_)])
                detections.append(detections_loc)

    detections = np.array(detections)
    gt = np.array(gt)

    e.evaluate_model(detections, gt)
