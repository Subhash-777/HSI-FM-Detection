import torch, numpy as np, sys, json
sys.path.insert(0, '.')
from src.models.litenet import LiteNet

device = torch.device('cuda')
model = LiteNet(n_bands=204, spatial_channels=3, spectral_output_dim=128,
                spatial_output_dim=64, fusion_hidden_dim=64, dropout=0.3,
                spectral_architecture='simple', spatial_input_size=3)
ck = torch.load('experiments/phase2_real/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

data = torch.load('data/processed/harmonized_204bands/agrifood_val_preprocessed.pt', map_location='cpu', weights_only=False)
spec = data['spectra'].float()
spat = data['spatial'].float()
labels = data['labels'].float().numpy()

bs = 256
all_probs = []
with torch.no_grad():
    for i in range(0, len(spec), bs):
        s = spec[i:i+bs].to(device)
        p = spat[i:i+bs].to(device)

        p1 = torch.sigmoid(model(s, p)).cpu().squeeze()
        p2 = torch.sigmoid(model(s, torch.flip(p, dims=[2]))).cpu().squeeze()
        p3 = torch.sigmoid(model(s, torch.flip(p, dims=[1]))).cpu().squeeze()
        p4 = torch.sigmoid(model(s, torch.rot90(p, k=1, dims=(1,2)))).cpu().squeeze()
        avg = (p1 + p2 + p3 + p4) / 4.0
        all_probs.append(avg.numpy())

probs = np.concatenate(all_probs)
lbl = labels.astype(int)

print(f'TTA Prob stats: min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}')
print()
print(f'Threshold   F1         Precision  Recall     IoU')
print('-' * 60)
best = {'thresh': 0, 'f1': 0}
for t in np.arange(0.30, 0.71, 0.025):
    pred = (probs >= t).astype(int)
    tp = np.sum((pred==1)&(lbl==1)); fp = np.sum((pred==1)&(lbl==0))
    fn = np.sum((pred==0)&(lbl==1))
    pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1)
    f1 = 2*pr*rc/max(pr+rc,1e-6)
    iou = tp/max(tp+fp+fn,1)
    marker = ' <---' if f1 > best['f1'] else ''
    if f1 > best['f1']: best = {'thresh': round(float(t),3), 'f1': round(f1,4), 'p': round(pr,4), 'r': round(rc,4), 'iou': round(iou,4)}
    print(f'{t:.3f}       {f1:.4f}     {pr:.4f}     {rc:.4f}     {iou:.4f}{marker}')

print()
print(f'BEST THRESHOLD: {best["thresh"]}')
print(f'  F1: {best["f1"]} | P: {best["p"]} | R: {best["r"]} | IoU: {best["iou"]}')

with open('results/tta_threshold_search.json', 'w') as f:
    json.dump(best, f, indent=2)
print()
print('Saved to results/tta_threshold_search.json')
