# Agri-SAR-ViT-SAR-Image-Colorization-in-Precision-Agriculture


Agri-SAR-ViT is a hybrid deep-learning architecture that translates dual-polarized (VV, VH) Sentinel-1 SAR data into high-fidelity, optical-equivalent RGB imagery. It bridges the "interpretability gap" of radar data for non-experts, enabling real-time crop and flood monitoring even in cloud-persistent regions.


## Key Features
**Hybrid Generator**: Combines Convolutional layers for local features with a **Swin Transformer bottleneck** for long-range spatial reasoning.
**Adversarial Refinement**: Utilizes a **PatchGAN discriminator** to enhance textural realism and reduce blur.
**Perceptual Optimization**: Guided by a composite loss (GAN + L1 + SSIM) and LPIPS for human-perceptual similarity.

## Performance Metrics
Our model outperforms baseline models (ResNet-UNet, Pix2Pix, CycleGAN):
| Metric | Value |
| :--- | :--- |
| **PSNR** | 31.24 dB  |
| **SSIM** | 0.922  |


## Installation & Usage
1. Clone the repo: `git clone https://github.com/your-username/Agri-SAR-ViT.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run deployment/app.py` 
