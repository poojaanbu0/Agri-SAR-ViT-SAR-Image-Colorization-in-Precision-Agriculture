# Agri-SAR-ViT-SAR-Image-Colorization-in-Precision-Agriculture


[cite_start]Agri-SAR-ViT is a hybrid deep-learning architecture that translates dual-polarized (VV, VH) Sentinel-1 SAR data into high-fidelity, optical-equivalent RGB imagery[cite: 17]. [cite_start]It bridges the "interpretability gap" of radar data for non-experts, enabling real-time crop and flood monitoring even in cloud-persistent regions[cite: 24, 34].


## Key Features
* [cite_start]**Hybrid Generator**: Combines Convolutional layers for local features with a **Swin Transformer bottleneck** for long-range spatial reasoning[cite: 135, 143].
* [cite_start]**Adversarial Refinement**: Utilizes a **PatchGAN discriminator** to enhance textural realism and reduce blur[cite: 146, 148].
* [cite_start]**Perceptual Optimization**: Guided by a composite loss (GAN + L1 + SSIM) and LPIPS for human-perceptual similarity[cite: 22, 204].

## Performance Metrics
[cite_start]Our model outperforms baseline models (ResNet-UNet, Pix2Pix, CycleGAN)[cite: 381]:
| Metric | Value |
| :--- | :--- |
| **PSNR** | [cite_start]31.24 dB [cite: 23] |
| **SSIM** | [cite_start]0.922 [cite: 23] |


## Installation & Usage
1. Clone the repo: `git clone https://github.com/your-username/Agri-SAR-ViT.git`
2. Install dependencies: `pip install -r requirements.txt`
3. [cite_start]Run the dashboard: `streamlit run deployment/app.py` [cite: 102]
