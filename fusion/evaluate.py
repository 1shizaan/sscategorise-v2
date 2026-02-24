# fusion/evaluate.py — generates the comparison chart 
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import matplotlib.pyplot as plt 
from fusion.classifier import train_fusion_classifier 
from vision.evaluate import run_evaluation as vision_accuracy 
 
def compare_and_chart(): 
    img_only_acc  = vision_accuracy()          # from Vision Engineer's module 
    multimodal_acc = train_fusion_classifier() * 100 
    labels  = ['Image-Only\n(EfficientNet)', 'Multimodal\n(Image + OCR + Fusion)'] 
    values  = [img_only_acc, multimodal_acc] 
    colors  = ['#93C5FD', '#1D4ED8'] 
    fig, ax = plt.subplots(figsize=(8, 5)) 
    bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor='white') 
    ax.set_ylim(0, 110) 
    ax.set_ylabel('Accuracy (%)', fontsize=13) 
    ax.set_title('SS Categorise v1 vs v2 Accuracy', fontsize=15, fontweight='bold') 
    for bar, val in zip(bars, values): 
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, 
fontweight='bold') 
    plt.tight_layout() 
    out = os.path.join(os.path.dirname(__file__), 'accuracy_comparison.png') 
    plt.savefig(out, dpi=150) 
    print(f'Chart saved to {out}') 
 
if __name__ == '__main__': 
    compare_and_chart() 
