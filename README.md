# WGAN - Wasserstein GAN

PyTorch implementation of WGAN (Wasserstein GAN) based on the paper [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Arjovsky et al. This implementation uses the DCGAN architecture with Wasserstein loss instead of the traditional GAN loss.

## Overview

Wasserstein GAN is an alternative to traditional GAN training that improves stability of learning, reduces mode collapse problems, and provides meaningful learning curves useful for debugging and hyperparameter searches. Unlike standard GANs that use Jensen-Shannon divergence, WGAN uses the Earth Mover's distance (Wasserstein-1 distance) as the loss function.

## Key Differences from Standard DCGAN

### 1. **Critic Instead of Discriminator**
- Removes the sigmoid activation from the final layer
- The network outputs raw scores instead of probabilities
- Called a "critic" because it learns to approximate the Wasserstein distance

### 2. **Wasserstein Loss**
- **Critic Loss**: `-(E[critic(real)] - E[critic(fake)])`
- **Generator Loss**: `-E[critic(fake)]`
- No logarithms or BCE loss

### 3. **Weight Clipping**
- Critic weights are clipped to [-0.01, 0.01] after each update
- Enforces the Lipschitz constraint required by Wasserstein distance

### 4. **RMSprop Optimizer**
- Uses RMSprop instead of Adam
- Lower learning rate (5e-5)

### 5. **Multiple Critic Iterations**
- Trains critic 5 times for every generator update
- WGAN allows training the critic to optimality, which provides better gradients for the generator

## Architecture

### Generator
- **Input**: 128-dimensional noise vector (z_dim=128)
- **Architecture**: Same as DCGAN
  - Transposed convolutions with stride=2
  - Batch normalization in all layers except output
  - ReLU activations
  - Tanh output activation
- **Output**: 64x64 images

### Critic (Modified Discriminator)
- **Input**: 64x64 images
- **Architecture**: Modified DCGAN discriminator
  - Strided convolutions (stride=2)
  - Instance normalization instead of batch normalization
  - LeakyReLU(0.2) activations
  - **No sigmoid at output** (key difference)
- **Output**: Raw scalar score

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
tensorboard
tqdm
```

## Installation

```bash
pip install torch torchvision tensorboard tqdm
```

## Usage

### Training

```bash
python train.py
```

### Testing Architecture

```bash
python model.py
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir=logs
```

Open `http://localhost:6006` in your browser to view training progress.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-5 | Lower than standard DCGAN |
| Batch Size | 64 | Smaller batch for stability |
| Image Size | 64x64 | Output resolution |
| Noise Dimension | 128 | Increased from 100 |
| Epochs | 5 | Number of training epochs |
| Critic Iterations | 5 | Critic updates per generator update |
| Weight Clip | 0.01 | Clip range for critic weights |
| Optimizer | RMSprop | Instead of Adam |

## Training Procedure

```python
for epoch in epochs:
    for batch in dataloader:
        # Train Critic multiple times
        for _ in range(CRITIC_ITERATIONS):
            1. Generate fake images
            2. Compute critic scores for real and fake
            3. Compute Wasserstein loss
            4. Update critic
            5. Clip critic weights to [-0.01, 0.01]
        
        # Train Generator once
        1. Generate fake images
        2. Compute critic score for fakes
        3. Compute generator loss
        4. Update generator
```

## Loss Functions

### Critic Loss (Wasserstein Distance)
```python
loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
```

The critic tries to maximize the difference between scores for real and fake images.

### Generator Loss
```python
loss_gen = -torch.mean(critic(fake))
```

The generator tries to maximize the critic's score for fake images.

## Key Implementation Details

### 1. Weight Clipping
```python
for p in critic.parameters():
    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
```

Applied after every critic update to enforce Lipschitz constraint.

### 2. Instance Normalization in Critic
Uses `nn.InstanceNorm2d` instead of `nn.BatchNorm2d` in the critic for better stability.

### 3. retain_graph=True
```python
loss_critic.backward(retain_graph=True)
```

Required because the same fake images are used for both critic and generator training within the same iteration.

### 4. Batch Normalization in Generator
Uses standard `nn.BatchNorm2d` in generator (unlike critic).

## Advantages of WGAN

✅ **More Stable Training**: Less sensitive to hyperparameters and architecture choices

✅ **No Mode Collapse**: WGAN experiments show no evidence of mode collapse

✅ **Meaningful Loss Curves**: Loss values correlate with sample quality

✅ **No Balance Required**: Can train critic to optimality without breaking training

✅ **Better Gradients**: Critic provides useful gradients even when well-trained

## Dataset

**Default**: MNIST (1 channel, grayscale)
- Auto-downloads on first run
- Resized to 64x64
- Normalized to [-1, 1]

**Alternative**: CelebA or custom datasets
```python
# Uncomment in train.py:
dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
# Change CHANNELS_IMG to 3 for RGB
```

## Project Structure

```
.
├── model.py           # Critic and Generator architecture
├── train.py           # WGAN training script
├── README.md          # This file
├── dataset/           # MNIST dataset (auto-downloaded)
└── logs/              # TensorBoard logs
    ├── real/          # Real images
    └── fake/          # Generated images
```

## Monitoring Training

WGAN provides more interpretable loss curves compared to standard GANs:

- **Critic Loss**: Should decrease and stabilize (becomes negative)
- **Generator Loss**: Should decrease (becomes more negative)
- Both losses should be meaningful indicators of training progress

## Common Issues & Solutions

### Issue: Poor Sample Quality
**Solutions**:
- Increase `CRITIC_ITERATIONS` (try 10)
- Decrease learning rate (try 1e-5)
- Train for more epochs
- Adjust `WEIGHT_CLIP` (try 0.05)

### Issue: Training Instability
**Solutions**:
- Ensure weight clipping is applied correctly
- Check that sigmoid is removed from critic output
- Verify RMSprop optimizer is being used
- Reduce learning rate

### Issue: Mode Collapse
WGAN is designed to avoid mode collapse, but if it occurs:
- Increase critic training iterations
- Adjust weight clip value
- Check normalization layers

## Limitations & Known Issues

⚠️ **Weight Clipping Drawbacks**: 
Weight clipping can lead to optimization difficulties and capacity underuse in the critic

**Alternative**: Consider implementing WGAN-GP (Gradient Penalty) which replaces weight clipping with a gradient penalty term for better results.

## Comparison: WGAN vs Standard GAN

| Aspect | Standard GAN | WGAN |
|--------|-------------|------|
| Loss Function | BCE (log loss) | Wasserstein distance |
| Discriminator Output | Probability [0,1] | Raw score (unbounded) |
| Optimizer | Adam | RMSprop |
| Learning Rate | 2e-4 | 5e-5 |
| D/G Balance | Critical | Less critical |
| Mode Collapse | Common issue | Rare |
| Loss Meaning | Not interpretable | Correlates with quality |
| Weight Constraint | None | Clipping to [-0.01, 0.01] |

## Tips for Better Results

1. **Train Longer**: WGAN benefits from extended training (20-100 epochs)
2. **Critic Iterations**: Experiment with 5-10 iterations per generator update
3. **Learning Rate**: Start with 5e-5, adjust based on loss curves
4. **Weight Clip**: Try values between 0.01-0.05
5. **Architecture**: Can use simpler architectures than standard GANs
6. **Monitor Losses**: Unlike standard GANs, WGAN losses are meaningful metrics

## Advanced: Upgrading to WGAN-GP

For even better results, consider implementing WGAN with Gradient Penalty:
- Replace weight clipping with gradient penalty
- Add penalty term: `λ * (||∇critic(interpolated)||₂ - 1)²`
- Use λ = 10 typically
- Remove weight clipping code

## References

- [Wasserstein GAN Paper](https://arxiv.org/abs/1701.07875) - Arjovsky et al., 2017
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) - Gulrajani et al., 2017 (WGAN-GP)
- [DCGAN Paper](https://arxiv.org/abs/1511.06434) - Radford et al., 2015

## Author

Programmed by Aladdin Persson
- Email: aladdin.persson@hotmail.com
- Initial coding: 2020-11-01
- Revision: 2022-12-20

## Citation

If you use this implementation, consider citing the original papers:

```bibtex
@article{arjovsky2017wasserstein,
  title={Wasserstein GAN},
  author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  journal={arXiv preprint arXiv:1701.07875},
  year={2017}
}
```

## License

Educational purposes. Check original paper licenses for research/commercial use.
