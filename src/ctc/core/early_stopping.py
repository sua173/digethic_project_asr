"""
Early Stopping implementation for training
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    
    Args:
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in the monitored quantity to qualify as an improvement
        mode: One of {'min', 'max'}. In 'min' mode, training will stop when the 
              monitored quantity stops decreasing; in 'max' mode it will stop when 
              the monitored quantity stops increasing
        verbose: If True, prints a message for each improvement
        baseline: Baseline value for the monitored metric. Training will stop if the
                 model doesn't show improvement over the baseline
        restore_best_weights: Whether to restore model weights from the epoch with the
                            best value of the monitored metric
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0001,
                 mode: str = 'min',
                 verbose: bool = True,
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True):
        
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode {mode} is unknown, please choose 'min' or 'max'")
        
        self.mode = mode
        
        # Initialize internal variables
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        
        # Set comparison functions based on mode
        if self.mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, score: float, model=None, epoch: int = 0) -> bool:
        """
        Check if early stopping criteria is met.
        
        Args:
            score: Current score to be monitored
            model: Model instance (optional, needed if restore_best_weights=True)
            epoch: Current epoch number
            
        Returns:
            True if training should be stopped, False otherwise
        """
        # First epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict()
            return False
        
        # Check for improvement
        if self.monitor_op(score - self.min_delta, self.best_score):
            # Improvement found
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict()
            
            if self.verbose:
                print(f"EarlyStopping: Improvement found at epoch {epoch+1}, "
                      f"best score: {self.best_score:.6f}")
        else:
            # No improvement
            self.counter += 1
            
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs "
                      f"(best: {self.best_score:.6f} at epoch {self.best_epoch+1})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training! "
                          f"Best score: {self.best_score:.6f} at epoch {self.best_epoch+1}")
                
                # Restore best weights if requested
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"EarlyStopping: Restored best weights from epoch {self.best_epoch+1}")
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping counter and state"""
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    @property
    def has_improved(self) -> bool:
        """Check if the last call resulted in an improvement"""
        return self.counter == 0 and self.best_score is not None


def test_early_stopping():
    """Test early stopping functionality"""
    
    print("Testing Early Stopping...")
    
    # Test 1: Min mode (loss)
    print("\n1. Testing min mode (loss):")
    early_stop = EarlyStopping(patience=3, mode='min', verbose=True)
    
    scores = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83, 0.84]  # Improvement stops at 0.8
    
    for epoch, score in enumerate(scores):
        should_stop = early_stop(score, epoch=epoch)
        if should_stop:
            print(f"Stopped at epoch {epoch}")
            break
    
    # Test 2: Max mode (accuracy)
    print("\n2. Testing max mode (accuracy):")
    early_stop = EarlyStopping(patience=2, mode='max', verbose=True)
    
    scores = [0.7, 0.8, 0.85, 0.84, 0.83]  # Improvement stops at 0.85
    
    for epoch, score in enumerate(scores):
        should_stop = early_stop(score, epoch=epoch)
        if should_stop:
            print(f"Stopped at epoch {epoch}")
            break
    
    # Test 3: With min_delta
    print("\n3. Testing with min_delta:")
    early_stop = EarlyStopping(patience=2, mode='min', min_delta=0.01, verbose=True)
    
    scores = [1.0, 0.995, 0.991, 0.988]  # Small improvements below min_delta
    
    for epoch, score in enumerate(scores):
        should_stop = early_stop(score, epoch=epoch)
        if should_stop:
            print(f"Stopped at epoch {epoch}")
            break
    
    print("\n✅ Early stopping tests completed!")


if __name__ == "__main__":
    test_early_stopping()