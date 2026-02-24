# vision/train.py 
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import torch 
import torchvision.transforms as T 
import torchvision.models as models 
import torch.nn as nn, torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.data import DataLoader, random_split 
from torchvision.datasets import ImageFolder 
from PIL import Image 
from tqdm import tqdm 
from config import DATASET_PATH, VISION_MODEL_PATH, NUM_CLASSES, BATCH_SIZE 
 
def pil_loader_rgb(path): 
    with open(path, 'rb') as f: 
        img = Image.open(f) 
        if img.mode in ('P', 'RGBA'): img = img.convert('RGBA') 
        return img.convert('RGB') 
 
def get_data_loaders(train_ratio=0.8): 
    train_tf = T.Compose([ 
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(), 
        T.ColorJitter(0.3, 0.3, 0.3, 0.1), T.RandAugment(), 
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) 
    ]) 
    val_tf = T.Compose([ 
        T.Resize(256), T.CenterCrop(224), 
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) 
    ]) 
    dataset = ImageFolder(root=DATASET_PATH, transform=train_tf, loader=pil_loader_rgb) 
    t = int(train_ratio * len(dataset)) 
    train_ds, val_ds = random_split(dataset, [t, len(dataset)-t]) 
    val_ds.dataset.transform = val_tf 
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0), 
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)) 
 
def build_model(): 
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) 
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES) 
    for p in m.parameters(): p.requires_grad = True 
    return m 
 
def train(num_epochs=10): 
    train_loader, val_loader = get_data_loaders() 
    model = build_model() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device) 
    print(f'Training on: {device}') 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.00005) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1) 
    best_val = float('inf') 
    for epoch in range(num_epochs): 
        model.train() 
        train_loss = 0 
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'): 
            imgs, labels = imgs.to(device), labels.to(device) 
            optimizer.zero_grad() 
            loss = criterion(model(imgs), labels) 
            loss.backward(); optimizer.step() 
            train_loss += loss.item() 
        model.eval(); val_loss = correct = total = 0 
        with torch.no_grad(): 
            for imgs, labels in val_loader: 
                imgs, labels = imgs.to(device), labels.to(device) 
                out = model(imgs) 
                val_loss += criterion(out, labels).item() 
                _, pred = torch.max(out, 1) 
                total += labels.size(0) 
                correct += (pred == labels).sum().item() 
        avg_val = val_loss / len(val_loader) 
        print(f'Val Loss: {avg_val:.4f} | Accuracy: {100*correct/total:.2f}%') 
        scheduler.step(avg_val) 
        if avg_val < best_val: 
            best_val = avg_val 
            torch.save(model.state_dict(), VISION_MODEL_PATH) 
            print(f'Saved best model to {VISION_MODEL_PATH}') 
    print('Training complete!') 
 
if __name__ == '__main__': 
    train()
