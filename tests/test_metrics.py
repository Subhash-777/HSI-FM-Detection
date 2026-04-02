"""
Test Metrics Module
Tests for evaluation metrics
"""

import pytest
import numpy as np
import torch

from src.training.metrics import PixelMetrics, ObjectMetrics
from src.training.losses import DiceLoss, FocalLoss, CombinedLoss
from src.evaluation.pixel_metrics import PixelLevelEvaluator
from src.evaluation.object_metrics import ObjectLevelEvaluator


class TestPixelMetrics:
    """Test PixelMetrics class"""
    
    def test_initialization(self):
        """Test metrics initialization"""
        metrics = PixelMetrics(threshold=0.5)
        assert metrics.threshold == 0.5
    
    def test_update_and_compute(self):
        """Test metric update and computation"""
        metrics = PixelMetrics()
        
        # Perfect predictions
        predictions = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['f1_score'] == 1.0
        assert results['iou'] == 1.0
    
    def test_imperfect_predictions(self):
        """Test with imperfect predictions"""
        metrics = PixelMetrics()
        
        predictions = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        # 2 TP, 1 FP, 0 FN, 1 TN
        assert results['precision'] == 2/3
        assert results['recall'] == 1.0
        assert 0 < results['f1_score'] < 1.0
    
    def test_reset(self):
        """Test metric reset"""
        metrics = PixelMetrics()
        
        predictions = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        
        metrics.update(predictions, targets)
        metrics.reset()
        
        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0
    
    def test_batch_updates(self):
        """Test multiple batch updates"""
        metrics = PixelMetrics()
        
        for _ in range(5):
            predictions = torch.rand(100) > 0.5
            targets = torch.rand(100) > 0.5
            metrics.update(predictions.float(), targets.float())
        
        results = metrics.compute()
        
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results


class TestObjectMetrics:
    """Test ObjectMetrics class"""
    
    def test_initialization(self):
        """Test metrics initialization"""
        metrics = ObjectMetrics(iou_threshold=0.5)
        assert metrics.iou_threshold == 0.5
    
    def test_perfect_detection(self):
        """Test with perfect object detection"""
        metrics = ObjectMetrics()
        
        # Single object, perfectly detected
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[40:60, 40:60] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[40:60, 40:60] = 1
        
        metrics.update(pred_mask, gt_mask)
        results = metrics.compute()
        
        assert results['detection_rate'] == 1.0
        assert results['total_objects'] == 1
        assert results['detected_objects'] == 1
    
    def test_partial_detection(self):
        """Test with partial overlap"""
        metrics = ObjectMetrics(iou_threshold=0.5)
        
        # Partial overlap
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[40:60, 40:60] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[45:65, 45:65] = 1
        
        metrics.update(pred_mask, gt_mask)
        results = metrics.compute()
        
        # Should be detected if IoU > 0.5
        assert results['total_objects'] == 1
    
    def test_multiple_objects(self):
        """Test with multiple objects"""
        metrics = ObjectMetrics()
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:20, 10:20] = 1
        pred_mask[50:60, 50:60] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:20, 10:20] = 1
        gt_mask[50:60, 50:60] = 1
        
        metrics.update(pred_mask, gt_mask)
        results = metrics.compute()
        
        assert results['total_objects'] == 2
        assert results['detected_objects'] == 2
    
    def test_reset(self):
        """Test metrics reset"""
        metrics = ObjectMetrics()
        
        pred_mask = np.ones((100, 100), dtype=np.uint8)
        gt_mask = np.ones((100, 100), dtype=np.uint8)
        
        metrics.update(pred_mask, gt_mask)
        metrics.reset()
        
        assert len(metrics.results) == 0


class TestLossFunctions:
    """Test loss functions"""
    
    def test_dice_loss(self, device):
        """Test Dice loss"""
        loss_fn = DiceLoss().to(device)
        
        predictions = torch.tensor([[0.9], [0.1], [0.8], [0.2]]).to(device)
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]]).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        assert loss.item() <= 1
    
    def test_dice_loss_perfect(self, device):
        """Test Dice loss with perfect predictions"""
        loss_fn = DiceLoss().to(device)
        
        predictions = torch.ones(10, 1).to(device)
        targets = torch.ones(10, 1).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() < 0.01  # Near zero
    
    def test_focal_loss(self, device):
        """Test Focal loss"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)
        
        predictions = torch.tensor([[0.9], [0.1], [0.8], [0.2]]).to(device)
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]]).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
    
    def test_combined_loss(self, device):
        """Test combined loss"""
        loss_fn = CombinedLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5
        ).to(device)
        
        predictions = torch.rand(16, 1).to(device)
        targets = (torch.rand(16, 1) > 0.5).float().to(device)
        
        total_loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'bce' in loss_dict
        assert 'dice' in loss_dict
        assert 'focal' in loss_dict
        assert 'total' in loss_dict
        
        assert total_loss.item() >= 0
    
    def test_loss_backward(self, device):
        """Test loss backward pass"""
        loss_fn = DiceLoss().to(device)
        
        predictions = torch.rand(8, 1, requires_grad=True).to(device)
        targets = torch.rand(8, 1).to(device)
        
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None


class TestPixelLevelEvaluator:
    """Test PixelLevelEvaluator class"""
    
    def test_initialization(self):
        """Test evaluator initialization"""
        evaluator = PixelLevelEvaluator(thresholds=[0.3, 0.5, 0.7])
        assert len(evaluator.thresholds) == 3
    
    def test_add_batch(self):
        """Test adding batch of predictions"""
        evaluator = PixelLevelEvaluator()
        
        predictions = np.random.rand(100, 100)
        targets = (np.random.rand(100, 100) > 0.5).astype(np.uint8)
        probabilities = np.random.rand(100, 100)
        
        evaluator.add_batch(predictions, targets, probabilities)
        
        assert len(evaluator.predictions) > 0
        assert len(evaluator.targets) > 0
        assert len(evaluator.probs) > 0
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        evaluator = PixelLevelEvaluator()
        
        # Perfect predictions
        predictions = np.ones(1000)
        targets = np.ones(1000)
        
        evaluator.add_batch(predictions, targets)
        metrics = evaluator.compute_metrics()
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_compute_at_thresholds(self):
        """Test metrics at multiple thresholds"""
        evaluator = PixelLevelEvaluator(thresholds=[0.3, 0.5, 0.7])
        
        probabilities = np.random.rand(1000)
        targets = (np.random.rand(1000) > 0.5).astype(np.uint8)
        
        evaluator.add_batch(None, targets, probabilities)
        results = evaluator.compute_metrics_at_thresholds()
        
        assert 'threshold_0.30' in results
        assert 'threshold_0.50' in results
        assert 'threshold_0.70' in results


class TestObjectLevelEvaluator:
    """Test ObjectLevelEvaluator class"""
    
    def test_initialization(self):
        """Test evaluator initialization"""
        evaluator = ObjectLevelEvaluator(iou_thresholds=[0.3, 0.5, 0.7])
        assert len(evaluator.iou_thresholds) == 3
    
    def test_add_sample(self):
        """Test adding sample"""
        evaluator = ObjectLevelEvaluator()
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:20, 10:20] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:20, 10:20] = 1
        
        evaluator.add_sample(pred_mask, gt_mask, sample_id="test_001")
        
        assert len(evaluator.results) == 1
    
    def test_compute_detection_rate(self):
        """Test detection rate computation"""
        evaluator = ObjectLevelEvaluator()
        
        # Perfect detection
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:20, 10:20] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:20, 10:20] = 1
        
        evaluator.add_sample(pred_mask, gt_mask)
        metrics = evaluator.compute_detection_rate(iou_threshold=0.5)
        
        assert metrics['detection_rate'] == 1.0
    
    def test_compute_average_iou(self):
        """Test average IoU computation"""
        evaluator = ObjectLevelEvaluator()
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:20, 10:20] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:20, 10:20] = 1
        
        evaluator.add_sample(pred_mask, gt_mask)
        avg_iou = evaluator.compute_average_iou()
        
        assert avg_iou == 1.0
    
    def test_compute_at_thresholds(self):
        """Test metrics at multiple IoU thresholds"""
        evaluator = ObjectLevelEvaluator(iou_thresholds=[0.3, 0.5, 0.7])
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:20, 10:20] = 1
        
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:20, 10:20] = 1
        
        evaluator.add_sample(pred_mask, gt_mask)
        results = evaluator.compute_metrics_at_thresholds()
        
        assert 'iou_0.3' in results
        assert 'iou_0.5' in results
        assert 'iou_0.7' in results
        assert 'average_iou' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
