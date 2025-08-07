# Enhanced Extreme Noise Denoising with SPOD-CNN

Deep learning framework for extreme thermal image denoising (90% salt-pepper + Gaussian noise) achieving 30+ dB PSNR.

## Installation

```bash
pip install torch torchvision opencv-python numpy scipy scikit-image matplotlib tqdm
```

## Dataset Structure

```
dataset/
├── train/clean/    # Clean images
├── train/noisy/    # Noisy images  
├── val/clean/
├── val/noisy/
├── test/clean/
└── test/noisy/
```

## Usage

```python
from enhanced_denoising import main

# Basic usage
results = main()

# Custom configuration
config = {
    'data_dir': '/path/to/dataset',
    'save_dir': '/path/to/results',
    'batch_size': 48,
    'epochs': 100
}
```

## Performance

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Single Model | 28.5 | 0.812 |
| 16-fold TTA | 30.1 | 0.847 |
| Ensemble | 30.6 | 0.852 |

## Key Features

- SPOD-CNN with complex-valued processing
- 12 architectural improvements for extreme noise
- Adaptive training with plateau restarts
- 16-fold test-time augmentation
- Comprehensive performance analysis

## Troubleshooting

- **Out of Memory**: Reduce batch_size
- **Dataset Not Found**: Check directory structure
- **Slow Training**: Increase num_workers or enable mixed precision

## License

MIT License
