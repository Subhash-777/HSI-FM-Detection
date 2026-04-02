"""
Test Models Module
Tests for neural network architectures
"""

import pytest
import torch
import torch.nn as nn

from src.models.spectral_branch import SpectralBranch1D, SpectralUNet1D
from src.models.spatial_branch import SpatialBranch2D
from src.models.fusion_net import FusionNet
from src.models.litenet import LiteNet


class TestSpectralBranch1D:
    """Test SpectralBranch1D model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = SpectralBranch1D(n_bands=204, output_dim=64)
        
        assert model.n_bands == 204
        assert model.output_dim == 64
    
    def test_forward_pass_3d_input(self, device):
        """Test forward pass with 3D input"""
        model = SpectralBranch1D(n_bands=204, output_dim=64).to(device)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 204).to(device)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 64)
    
    def test_forward_pass_2d_input(self, device):
        """Test forward pass with 2D input"""
        model = SpectralBranch1D(n_bands=204, output_dim=64).to(device)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 204).to(device)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 64)
    
    def test_gradient_flow(self, device):
        """Test that gradients flow properly"""
        model = SpectralBranch1D(n_bands=204, output_dim=64).to(device)
        
        input_tensor = torch.randn(2, 1, 204, requires_grad=True).to(device)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None


class TestSpectralUNet1D:
    """Test SpectralUNet1D model"""
    
    def test_initialization(self):
        """Test U-Net initialization"""
        model = SpectralUNet1D(n_bands=204, output_dim=64)
        
        assert model.n_bands == 204
        assert model.output_dim == 64
    
    def test_forward_pass(self, device):
        """Test forward pass"""
        model = SpectralUNet1D(n_bands=204, output_dim=64).to(device)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 204).to(device)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 64)
    
    def test_skip_connections(self, device):
        """Test that skip connections work"""
        model = SpectralUNet1D(n_bands=204, output_dim=64).to(device)
        model.eval()
        
        input_tensor = torch.randn(2, 1, 204).to(device)
        
        # Should not raise any errors
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 64)


class TestSpatialBranch2D:
    """Test SpatialBranch2D model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = SpatialBranch2D(in_channels=3, output_dim=64)
        
        assert model.in_channels == 3
        assert model.output_dim == 64
    
    def test_forward_pass(self, device):
        """Test forward pass"""
        model = SpatialBranch2D(in_channels=3, output_dim=64).to(device)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 64)
    
    def test_different_input_sizes(self, device):
        """Test with different spatial sizes"""
        model = SpatialBranch2D(in_channels=3, output_dim=64).to(device)
        
        # Test with 128x128
        input_128 = torch.randn(2, 3, 128, 128).to(device)
        output_128 = model(input_128)
        assert output_128.shape == (2, 64)
        
        # Test with 64x64
        input_64 = torch.randn(2, 3, 64, 64).to(device)
        output_64 = model(input_64)
        assert output_64.shape == (2, 64)


class TestFusionNet:
    """Test FusionNet model"""
    
    def test_initialization(self):
        """Test fusion network initialization"""
        model = FusionNet(
            spectral_dim=64,
            spatial_dim=64,
            hidden_dim=64,
            dropout=0.3
        )
        
        assert model is not None
    
    def test_forward_pass(self, device):
        """Test forward pass"""
        model = FusionNet(
            spectral_dim=64,
            spatial_dim=64,
            hidden_dim=64
        ).to(device)
        
        batch_size = 4
        spectral_feat = torch.randn(batch_size, 64).to(device)
        spatial_feat = torch.randn(batch_size, 64).to(device)
        
        output = model(spectral_feat, spatial_feat)
        
        assert output.shape == (batch_size, 1)
        # Output should be in [0, 1] due to sigmoid
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_different_dimensions(self, device):
        """Test with different input dimensions"""
        model = FusionNet(
            spectral_dim=32,
            spatial_dim=64,
            hidden_dim=48
        ).to(device)
        
        spectral_feat = torch.randn(2, 32).to(device)
        spatial_feat = torch.randn(2, 64).to(device)
        
        output = model(spectral_feat, spatial_feat)
        
        assert output.shape == (2, 1)


class TestLiteNet:
    """Test complete LiteNet model"""
    
    def test_initialization(self):
        """Test LiteNet initialization"""
        model = LiteNet(
            n_bands=204,
            spatial_channels=3,
            spectral_architecture='simple'
        )
        
        assert model.n_bands == 204
        assert model.spatial_channels == 3
    
    def test_forward_pass(self, device):
        """Test forward pass"""
        model = LiteNet(n_bands=204).to(device)
        
        batch_size = 4
        spectrum = torch.randn(batch_size, 1, 204).to(device)
        spatial = torch.randn(batch_size, 3, 256, 256).to(device)
        
        output = model(spectrum, spatial)
        
        assert output.shape == (batch_size, 1)
        assert (output >= 0).all() and (output <= 1).all()
    
    def test_unet_architecture(self, device):
        """Test with U-Net spectral branch"""
        model = LiteNet(
            n_bands=204,
            spectral_architecture='unet'
        ).to(device)
        
        spectrum = torch.randn(2, 1, 204).to(device)
        spatial = torch.randn(2, 3, 256, 256).to(device)
        
        output = model(spectrum, spatial)
        
        assert output.shape == (2, 1)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        model = LiteNet(n_bands=204)
        
        param_count = model.count_parameters()
        
        assert param_count > 0
        # Should be lightweight (< 5M parameters)
        assert param_count < 5_000_000
    
    def test_get_feature_maps(self, device):
        """Test feature extraction"""
        model = LiteNet(n_bands=204).to(device)
        
        spectrum = torch.randn(2, 1, 204).to(device)
        spatial = torch.randn(2, 3, 256, 256).to(device)
        
        features = model.get_feature_maps(spectrum, spatial)
        
        assert 'spectral_features' in features
        assert 'spatial_features' in features
        assert features['spectral_features'].shape == (2, 64)
        assert features['spatial_features'].shape == (2, 64)
    
    def test_backward_pass(self, device):
        """Test backward pass"""
        model = LiteNet(n_bands=204).to(device)
        
        spectrum = torch.randn(2, 1, 204, requires_grad=True).to(device)
        spatial = torch.randn(2, 3, 256, 256, requires_grad=True).to(device)
        
        output = model(spectrum, spatial)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert spectrum.grad is not None
        assert spatial.grad is not None
    
    def test_eval_mode(self, device):
        """Test evaluation mode"""
        model = LiteNet(n_bands=204).to(device)
        model.eval()
        
        spectrum = torch.randn(2, 1, 204).to(device)
        spatial = torch.randn(2, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output1 = model(spectrum, spatial)
            output2 = model(spectrum, spatial)
        
        # Outputs should be deterministic in eval mode
        torch.testing.assert_close(output1, output2)


# Integration Tests
class TestModelIntegration:
    """Integration tests for complete model pipeline"""
    
    def test_end_to_end_inference(self, device):
        """Test end-to-end inference"""
        model = LiteNet(n_bands=204).to(device)
        model.eval()
        
        # Simulate batch of samples
        batch_size = 8
        spectrum = torch.randn(batch_size, 1, 204).to(device)
        spatial = torch.randn(batch_size, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = model(spectrum, spatial)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mixed_precision_compatibility(self, device):
        """Test compatibility with mixed precision training"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = LiteNet(n_bands=204).to(device)
        
        spectrum = torch.randn(2, 1, 204).to(device)
        spatial = torch.randn(2, 3, 256, 256).to(device)
        
        from torch.cuda.amp import autocast
        
        with autocast():
            output = model(spectrum, spatial)
        
        assert output.dtype == torch.float16
    
    def test_model_save_load(self, device, temp_dir):
        """Test model saving and loading"""
        model = LiteNet(n_bands=204).to(device)
        
        # Save model
        model_path = temp_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = LiteNet(n_bands=204).to(device)
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Compare outputs
        model.eval()
        loaded_model.eval()
        
        spectrum = torch.randn(2, 1, 204).to(device)
        spatial = torch.randn(2, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output1 = model(spectrum, spatial)
            output2 = loaded_model(spectrum, spatial)
        
        torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
