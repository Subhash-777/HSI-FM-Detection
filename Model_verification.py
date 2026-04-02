import torch
from src.models.litenet import LiteNet

# Load trained model
checkpoint = torch.load('experiments/phase1_synthetic/best_model.pth')
model = LiteNet(n_bands=204, spatial_input_size=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test with dummy data
spectrum = torch.randn(1, 204)
spatial = torch.randn(1, 3, 3, 3)

with torch.no_grad():
    logits = model(spectrum, spatial)
    prob = torch.sigmoid(logits)
    print(f"Prediction probability: {prob.item():.4f}")
    print("✓ Model inference works!")
