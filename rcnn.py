import os
import cv2
import gc
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import joblib
import xml.etree.ElementTree as ET
from collections import defaultdict

# ----------------------------- Config -----------------------------
DEVICE = (
    torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"[DEVICE] {DEVICE}")

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS_TO_IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

DATA_ROOT    = "./VOC2007"
JPEG_DIR     = os.path.join(DATA_ROOT, "JPEGImages")
ANNOT_DIR    = os.path.join(DATA_ROOT, "Annotations")
CACHE_DIR    = "./RCNNDataCache"
FEATURE_DIR  = os.path.join(CACHE_DIR, "features")
MODEL_DIR    = os.path.join(CACHE_DIR, "models")
RESULT_DIR   = os.path.join(CACHE_DIR, "results")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR,  exist_ok=True)

PROPOSAL_PATH = os.path.join(CACHE_DIR, "proposals.pkl")
INDEX_FILE    = os.path.join(CACHE_DIR, "rcnn_index.txt")
ALEXNET_PATH  = os.path.join(MODEL_DIR, "custom_alexnet.pth")

# ----------------------------- Selective Search -----------------------------
from cv2.ximgproc import segmentation
cv2.setNumThreads(1)
_ss = None
def init_ss():
    global _ss
    _ss = segmentation.createSelectiveSearchSegmentation()

def compute_proposals(img_id):
    img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
    if img is None:
        return img_id, []
    _ss.setBaseImage(img)
    _ss.switchToSelectiveSearchFast()
    rects = _ss.process()[:500]  # 최대 2000개
    filtered = [r for r in rects if r[2]>=20 and r[3]>=20]
    return img_id, filtered

def selective_search_all_images():
    ids = [f[:-4] for f in os.listdir(JPEG_DIR) if f.endswith(".jpg")]
    props = {}
    cores = max(1, mp.cpu_count()-2)
    with mp.Pool(cores, initializer=init_ss) as pool:
        for img_id, boxes in tqdm(pool.imap(compute_proposals, ids),
                                  total=len(ids),
                                  desc="Selective Search"):
            if boxes:
                props[img_id] = boxes
    return props

# ----------------------------- Index 생성 -----------------------------
def compute_iou(a, b):
    xa, ya, xa2, ya2 = a; xb, yb, xb2, yb2 = b
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw = max(ix2-ix1+1,0); ih = max(iy2-iy1+1,0)
    inter = iw*ih
    union = (xa2-xa+1)*(ya2-ya+1)+(xb2-xb+1)*(yb2-yb+1)-inter
    return inter/union if union>0 else 0

def generate_index_file(props, path):
    with open(path, "w") as f:
        for img_id, boxes in tqdm(props.items(), desc="Generate Index"):
            tree = ET.parse(os.path.join(ANNOT_DIR, f"{img_id}.xml"))
            gts = []
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                bb = obj.find("bndbox")
                xmin = int(bb.find("xmin").text)
                ymin = int(bb.find("ymin").text)
                xmax = int(bb.find("xmax").text)
                ymax = int(bb.find("ymax").text)
                gts.append((CLASS_TO_IDX[cls], (xmin, ymin, xmax, ymax)))
            for idx, (x,y,w,h) in enumerate(boxes):
                prop = (x, y, x+w, y+h)
                best_i, best_lbl = 0, 20
                for cls_i, gt_bb in gts:
                    iou = compute_iou(prop, gt_bb)
                    if iou > best_i:
                        best_i, best_lbl = iou, cls_i
                if best_i >= 0.5:
                    lbl = best_lbl
                elif best_i < 0.3:
                    lbl = 20
                else:
                    continue
                f.write(f"{img_id},{idx},{lbl},{x},{y},{w},{h}\n")

# ----------------------------- CustomAlexNet 정의 -----------------------------
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,11,4,2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(64,192,5,padding=2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(192,384,3,padding=1), nn.ReLU(),
            nn.Conv2d(384,256,3,padding=1), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256*6*6,4096), nn.ReLU(),
            nn.Dropout(), nn.Linear(4096,4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

    def extract_fc7(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        # fc6 & fc7
        return self.classifier[:4](x)

# ----------------------------- PatchDataset -----------------------------
crop_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class PatchDataset(Dataset):
    def __init__(self, index_file):
        self.entries = []
        with open(index_file) as f:
            for line in f:
                img_id, idx, lbl, x,y,w,h = line.strip().split(",")
                self.entries.append((img_id,int(lbl),int(x),int(y),int(w),int(h)))
    def __len__(self):
        return len(self.entries)
    def __getitem__(self,i):
        img_id, lbl, x,y,w,h = self.entries[i]
        img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
        roi = img[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(roi)
        tensor = crop_transform(pil)
        return tensor, lbl

# ----------------------------- Feature Extract -----------------------------
def extract_and_save_features(model, props, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for img_id, boxes in tqdm(props.items(), desc="Feat Extract"):
        img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
        if img is None: continue
        regions = []
        for idx, (x,y,w,h) in enumerate(boxes):
            roi = img[y:y+h, x:x+w]
            if roi.shape[0]<20 or roi.shape[1]<20: continue
            pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            tensor = crop_transform(pil).unsqueeze(0).to(DEVICE)
            regions.append((idx, tensor))
        if not regions: continue
        # 배치 단위로 처리
        for i in range(0, len(regions), 32):
            batch = torch.cat([t for _,t in regions[i:i+100]], dim=0)
            idxs  = [idx for idx,_ in regions[i:i+100]]
            with torch.no_grad():
                feats = model.extract_fc7(batch).cpu().numpy()  # (B,4096)
            for j, vec in enumerate(feats):
                np.save(os.path.join(out_dir, f"{img_id}_{idxs[j]}.npy"), vec)
        del regions, batch, feats; gc.collect()

# ----------------------------- train_model -----------------------------
def train_model():
    ds = PatchDataset(INDEX_FILE)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    model = CustomAlexNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(5):
        model.train()
        total, correct = 0, 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {ep+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss_fn(out, labels).backward()
            optimizer.step()
            correct += (out.argmax(1)==labels).sum().item()
            total += labels.size(0)
        print(f"[Epoch {ep+1}] Acc: {correct/total*100:.2f}%")
    torch.save(model.state_dict(), ALEXNET_PATH)
    return model

# ----------------------------- train_svms -----------------------------
def train_svms():
    idx_dict = {}
    with open(INDEX_FILE) as f:
        for line in f:
            img_id, prop_idx, lbl, *_ = line.strip().split(",")
            idx_dict[(img_id,int(prop_idx))] = int(lbl)
    for c, cls in enumerate(VOC_CLASSES):
        pkl = os.path.join(MODEL_DIR, f"svm_{cls}.pkl")
        if os.path.exists(pkl):
            print(f"[SVM] Load {cls}")
            continue
        Xp, Xn = [], []
        for fn in os.listdir(FEATURE_DIR):
            if not fn.endswith(".npy"): continue
            img_id, idx = fn[:-4].split("_")
            lbl = idx_dict.get((img_id,int(idx)))
            vec = np.load(os.path.join(FEATURE_DIR, fn))
            if lbl==c:   Xp.append(vec)
            elif lbl==20:Xn.append(vec)
        if not Xp or not Xn:
            print(f"[SVM] Skip {cls}")
            continue
        X = np.vstack([Xp, Xn])
        y = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])
        clf = LinearSVC(C=0.01, max_iter=10000)
        clf.fit(X, y)
        joblib.dump(clf, pkl)
        print(f"[SVM] Done {cls}")

# ----------------------------- train_bbox_regressors -----------------------------
def train_bbox_regressors():
    idx_dict = {}
    with open(INDEX_FILE) as f:
        for line in f:
            img_id, prop_idx, lbl, x,y,w,h = line.strip().split(",")
            idx_dict[(img_id,int(prop_idx))] = (int(lbl),int(x),int(y),int(w),int(h))
    for c, cls in enumerate(VOC_CLASSES):
        pkl = os.path.join(MODEL_DIR, f"reg_{cls}.pkl")
        if os.path.exists(pkl):
            print(f"[Reg] Load {cls}")
            continue
        X, Y = [], []
        for (img_id,prop_idx),(lbl,x,y,w,h) in idx_dict.items():
            if lbl!=c: continue
            tree = ET.parse(os.path.join(ANNOT_DIR, f"{img_id}.xml"))
            for obj in tree.findall("object"):
                if obj.find("name").text!=cls: continue
                bb = obj.find("bndbox")
                gx1,gy1 = int(bb.find("xmin").text), int(bb.find("ymin").text)
                gx2,gy2 = int(bb.find("xmax").text), int(bb.find("ymax").text)
                pw, ph = w, h
                tx, ty = (gx1 - x)/pw, (gy1 - y)/ph
                tw, th = np.log((gx2-gx1+1)/pw), np.log((gy2-gy1+1)/ph)
                vec = np.load(os.path.join(FEATURE_DIR, f"{img_id}_{prop_idx}.npy"))
                X.append(vec); Y.append([tx,ty,tw,th])
                break
        if not X: continue
        reg = LinearRegression()
        reg.fit(np.vstack(X), np.vstack(Y))
        joblib.dump(reg, pkl)
        print(f"[Reg] Done {cls}")

# ----------------------------- inference, nms, evaluate_map -----------------------------
def apply_bbox_regression(box, offs):
    x,y,w,h = box; tx,ty,tw,th = offs
    cx, cy = x+0.5*w, y+0.5*h
    cx_p,cy_p = tx*w+cx, ty*h+cy
    w_p, h_p = np.exp(tw)*w, np.exp(th)*h
    return [int(cx_p-0.5*w_p),int(cy_p-0.5*h_p),int(cx_p+0.5*w_p),int(cy_p+0.5*h_p)]

def inference():
    svms, regs = {}, {}
    for i, cls in enumerate(VOC_CLASSES):
        sp = os.path.join(MODEL_DIR, f"svm_{cls}.pkl")
        rp = os.path.join(MODEL_DIR, f"reg_{cls}.pkl")
        if os.path.exists(sp): svms[i]=joblib.load(sp)
        if os.path.exists(rp): regs[i]=joblib.load(rp)
    lines = open(INDEX_FILE).read().splitlines()
    dets=[]
    for fn in tqdm(os.listdir(FEATURE_DIR), desc="Inference"):
        if not fn.endswith(".npy"): continue
        img_id, prop_idx = fn[:-4].split("_")
        feat = np.load(os.path.join(FEATURE_DIR, fn)).reshape(1,-1)
        for L in lines:
            if L.startswith(f"{img_id},{prop_idx},"):
                _,_,_,x,y,w,h = L.split(","); box=(int(x),int(y),int(w),int(h))
                break
        for c, clf in svms.items():
            score = clf.decision_function(feat)[0]
            if score<=0: continue
            bbox = apply_bbox_regression(box, regs[c].predict(feat)[0])
            dets.append((img_id, c, score, *bbox))
    return dets

def nms(dets, iou_thr=0.3):
    keep=[]; dets=sorted(dets, key=lambda x:x[2], reverse=True)
    while dets:
        best=dets.pop(0); keep.append(best)
        dets=[d for d in dets if d[0]!=best[0] or compute_iou(d[3:], best[3:])<iou_thr]
    return keep

def evaluate_map(dets):
    gt=defaultdict(list)
    for f in os.listdir(ANNOT_DIR):
        img_id=f[:-4]
        tree=ET.parse(os.path.join(ANNOT_DIR, f))
        for obj in tree.findall("object"):
            bb=obj.find("bndbox")
            gt[img_id].append([
                int(bb.find("xmin").text),int(bb.find("ymin").text),
                int(bb.find("xmax").text),int(bb.find("ymax").text)
            ])
    by_cls=defaultdict(list)
    for img_id,c,score,x1,y1,x2,y2 in dets:
        by_cls[c].append((img_id,score,[x1,y1,x2,y2]))
    aps=[]
    for c, preds in by_cls.items():
        preds=sorted(preds, key=lambda x:-x[1])
        tp, fp, matched = [], [], set()
        total = sum(len(v) for v in gt.values())
        for img_id,_,bb in preds:
            found=False
            for i, gt_bb in enumerate(gt[img_id]):
                if compute_iou(bb, gt_bb)>0.5 and (img_id,i) not in matched:
                    tp.append(1); fp.append(0); matched.add((img_id,i)); found=True; break
            if not found:
                tp.append(0); fp.append(1)
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        rec = tp/total if total else tp*0
        prec= tp/np.maximum(tp+fp, np.finfo(float).eps)
        ap = sum((np.max(prec[rec>=t]) if np.any(rec>=t) else 0)/11.0 for t in np.arange(0,1.1,0.1))
        print(f"AP {VOC_CLASSES[c]} = {ap:.4f}")
        aps.append(ap)
    print(f"Final mAP = {np.mean(aps):.4f}")

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    # 1) proposals
    if os.path.exists(PROPOSAL_PATH):
        print("[1] \ub85c\ub4dc: proposals")
        with open(PROPOSAL_PATH,"rb") as f: props=pickle.load(f)
    else:
        print("[1] \uac1c\uc120: proposals")
        props=selective_search_all_images()
        with open(PROPOSAL_PATH,"wb") as f: pickle.dump(props,f)

    # 2) index
    if not os.path.exists(INDEX_FILE):
        print("[2] \uac1c\uc120: index")
        generate_index_file(props, INDEX_FILE)
    else:
        print("[2] \ub85c\ub4dc: index")

    # 3) train CustomAlexNet
    print("[3] \ud559\uc0dd: CustomAlexNet from scratch")
    model = train_model()

    # 4) feature extract
    if any(fn.endswith(".npy") for fn in os.listdir(FEATURE_DIR)):
        print("[4] \ub85c\ub4dc: features")
    else:
        print("[4] \uac1c\uc120: features")
        extract_and_save_features(model, props, FEATURE_DIR)

    # 5) SVM
    print("[5] SVM \ud559\uc0dd")
    train_svms()

    # 6) BBox regression
    print("[6] BBox \ud68c\uae30 \ud559\uc0dd")
    train_bbox_regressors()

    # 7) inference, NMS, mAP
    print("[7] \uac80\uc0ac: inference \u2192 NMS \u2192 mAP")
    raw = inference()
    final = nms(raw)
    evaluate_map(final)
