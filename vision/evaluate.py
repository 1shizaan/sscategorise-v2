# vision/evaluate.py 
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
from torch.utils.data import DataLoader 
from config import DATASET_PATH, NUM_CLASSES, CATEGORIES, IMAGE_SIZE 
from vision.model import load_model 
 
def run_evaluation(): 
    model = load_model() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device) 
    tf = transforms.Compose([ 
        transforms.Resize(256), transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) 
    ]) 
    dataset = ImageFolder(root=DATASET_PATH, transform=tf) 
    loader  = DataLoader(dataset, batch_size=16, shuffle=False) 
    all_preds, all_labels = [], [] 
    with torch.no_grad(): 
        for imgs, labels in loader: 
            out = model(imgs.to(device)) 
            _, preds = torch.max(out, 1) 
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.numpy()) 
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels)) 
    print(f'Image-Only Accuracy: {accuracy:.2f}%') 
    # Export confusion matrix 
    cm = confusion_matrix(all_labels, all_preds) 
    disp = ConfusionMatrixDisplay(cm, display_labels=dataset.classes) 
    fig, ax = plt.subplots(figsize=(14, 12)) 
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False) 
    plt.tight_layout() 
    out_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png') 
    plt.savefig(out_path, dpi=150) 
    print(f'Confusion matrix saved to {out_path}') 
    return accuracy 
 
if __name__ == '__main__': 
    run_evaluation() 
