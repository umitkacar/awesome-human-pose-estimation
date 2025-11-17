# 🧍 Awesome Human Pose Estimation [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<div align="center">

  <img src="https://img.shields.io/badge/Last%20Updated-2025--01-blue?style=for-the-badge" alt="Last Updated">
  <img src="https://img.shields.io/badge/Papers-200+-green?style=for-the-badge" alt="Papers">
  <img src="https://img.shields.io/badge/Datasets-25+-orange?style=for-the-badge" alt="Datasets">
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="License">

  <h3>🎯 A comprehensive collection of papers, datasets, tools, and resources for Human Pose Estimation</h3>

  <p><i>Covering 2D/3D Body, Hand, Face, Whole-Body Pose Estimation and more</i></p>

</div>

---

## 📑 Table of Contents

- [🌟 Introduction](#-introduction)
- [🎓 Survey Papers](#-survey-papers)
- [🧍 2D Human Pose Estimation](#-2d-human-pose-estimation)
  - [Top-Down Methods](#top-down-methods)
  - [Bottom-Up Methods](#bottom-up-methods)
  - [Transformer-Based Methods](#transformer-based-methods)
- [🎭 3D Human Pose Estimation](#-3d-human-pose-estimation)
  - [Monocular 3D Pose](#monocular-3d-pose)
  - [Multi-View 3D Pose](#multi-view-3d-pose)
  - [Video-Based 3D Pose](#video-based-3d-pose)
- [✋ Hand Pose Estimation](#-hand-pose-estimation)
- [😊 Face & Facial Landmark Detection](#-face--facial-landmark-detection)
- [👤 Whole-Body Pose Estimation](#-whole-body-pose-estimation)
- [👥 Multi-Person Pose Estimation](#-multi-person-pose-estimation)
- [⚡ Real-Time & Lightweight Models](#-real-time--lightweight-models)
- [📊 Datasets](#-datasets)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [🛠️ Tools & Libraries](#️-tools--libraries)
- [🎯 Applications](#-applications)
- [📚 Resources](#-resources)

---

## 🌟 Introduction

Human Pose Estimation (HPE) is the task of estimating the configuration of the body (pose) from an image or video. It involves detecting and localizing key anatomical points (keypoints) such as joints, hands, facial features, and connecting them to form a skeletal structure.

### Key Challenges
- 🔸 **Occlusion**: Body parts hidden by objects or other people
- 🔸 **Scale Variation**: People at different distances from camera
- 🔸 **Lighting Conditions**: Varying illumination and shadows
- 🔸 **Complex Poses**: Unusual body configurations
- 🔸 **Real-time Performance**: Balancing accuracy vs. speed
- 🔸 **Depth Ambiguity**: Inferring 3D from 2D images

---

## 🎓 Survey Papers

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | Two-Dimensional Human Pose Estimation with Deep Learning: A Review | Applied Sciences | [Paper](https://www.mdpi.com/2076-3417/15/13/7344) |
| 2025 | A systematic survey on human pose estimation | AI Review | [Paper](https://link.springer.com/article/10.1007/s10462-024-11060-2) |
| 2025 | A Survey of the State of the Art in Monocular 3D Human Pose Estimation | Sensors | [Paper](https://www.mdpi.com/1424-8220/25/8/2409) |
| 2024 | A survey on deep 3D human pose estimation | AI Review | [Paper](https://link.springer.com/article/10.1007/s10462-024-11019-3) |
| 2024 | Deep Learning for 3D Human Pose Estimation and Mesh Recovery | arXiv | [Paper](https://arxiv.org/abs/2402.18844) |
| 2023 | Deep Learning-based Human Pose Estimation: A Survey | ACM Computing Surveys | [Paper](https://dl.acm.com/doi/10.1145/3603618) |
| 2022 | Recent Advances of Monocular 2D and 3D HPE: A Deep Learning Perspective | ACM Computing Surveys | [Paper](https://dl.acm.com/doi/10.1145/3524497) |

---

## 🧍 2D Human Pose Estimation

2D HPE aims to detect human joint locations in pixel coordinates from RGB images.

### Top-Down Methods

Top-down methods first detect person bounding boxes, then estimate pose for each person.

#### 🔥 State-of-the-Art Methods (2024-2025)

| Method | Year | Venue | Key Features | Code |
|--------|------|-------|--------------|------|
| **VTTransPose** | 2024 | Scientific Reports | Efficient transformer-based 2D pose estimation | - |
| **CCAM-Person** | 2024 | Scientific Reports | YOLOv8-based real-time HPE | - |
| **ViTPose++** | 2023 | TPAMI | Vision transformer for generic body pose | [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |
| **HRNet** | 2020 | CVPR | High-resolution representations | [GitHub](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) |
| **SimpleBaseline** | 2018 | ECCV | Simple yet effective baseline | [GitHub](https://github.com/microsoft/human-pose-estimation.pytorch) |

#### 📝 Classic Methods

**HRNet (High-Resolution Network)**
- 🎯 Maintains high-resolution representations throughout
- 🎯 Parallel multi-resolution subnetworks
- 🎯 Repeated multi-scale fusion
- 📄 [Paper](https://arxiv.org/abs/1902.09212) | [Code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

**SimpleBaseline**
- 🎯 ResNet backbone + deconvolution layers
- 🎯 Simple architecture, strong performance
- 📄 [Paper](https://arxiv.org/abs/1804.06208) | [Code](https://github.com/microsoft/human-pose-estimation.pytorch)

**Hourglass Networks**
- 🎯 Stacked hourglass architecture
- 🎯 Bottom-up, top-down processing
- 📄 [Paper](https://arxiv.org/abs/1603.06937)

### Bottom-Up Methods

Bottom-up methods detect all keypoints first, then group them into individuals.

| Method | Year | Key Features | Code |
|--------|------|--------------|------|
| **OpenPose** | 2019 | Part Affinity Fields (PAFs), multi-person real-time | [GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |
| **HigherHRNet** | 2020 | Multi-resolution heatmaps | [GitHub](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) |
| **AssociativeEmbedding** | 2017 | Grouping via embeddings | [GitHub](https://github.com/princeton-vl/pose-ae-train) |

**OpenPose** 🌟
- First real-time multi-person 2D pose estimation
- Detects body (25 points), hands (21 points each), face (70 points)
- Uses Part Affinity Fields for limb association
- 📄 [Paper](https://arxiv.org/abs/1812.08008) | [Code](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Transformer-Based Methods

Latest approaches using vision transformers and attention mechanisms.

| Method | Year | Venue | Highlights | Code |
|--------|------|-------|------------|------|
| **ViTPose** | 2022 | NeurIPS | Simple ViT baselines, SOTA on COCO | [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |
| **ViTPose++** | 2023 | TPAMI | Generic body pose (human, animal, etc.) | [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |
| **TokenPose** | 2021 | ICCV | Token-based representation | [GitHub](https://github.com/leeyegy/TokenPose) |
| **PCT** | 2022 | CVPR | Pose transformer with convolutions | - |

**ViTPose Key Features:**
- ✅ Plain vision transformer encoder
- ✅ Scalable architecture (ViTPose-S/B/L/H)
- ✅ Shifted window & pooling attention
- ✅ COCO val: 81.1 AP (ViTPose-H)
- 📄 [Paper](https://arxiv.org/abs/2204.12484)

---

## 🎭 3D Human Pose Estimation

Estimating 3D joint locations in world coordinates from images or videos - the foundation for AR/VR, motion capture, sports analytics, and healthcare applications.

### 🎯 Problem Settings & Approaches

<table>
<tr>
<td width="33%">

**Monocular RGB** 🔥
- Single camera input
- Depth ambiguity challenge
- Most practical setting
- Latest: SSMs, Diffusion

</td>
<td width="33%">

**Multi-View**
- Multiple synchronized cameras
- Triangulation/voxel-based
- High accuracy
- Costly setup

</td>
<td width="33%">

**Video-Based**
- Temporal consistency
- Motion priors
- Smooth trajectories
- Real-world applicable

</td>
</tr>
</table>

---

### 🔥 Latest Methods (2024-2025)

#### State-Space Models (Mamba Architecture)

**SasMamba** (November 2024) - SOTA Efficiency
- Structure-Aware Stride SSM (SAS-SSM)
- Multi-scale global representations
- Linear complexity O(n)
- 5× faster than Transformers
- 📄 [Paper](https://arxiv.org/abs/2511.08872)

**PoseMamba** (August 2024)
- Bidirectional global-local spatio-temporal SSM
- Purely SSM-based (no convolutions)
- Linear complexity for long sequences
- Spatial reordering strategy
- 📄 [Paper](https://arxiv.org/abs/2408.03540)

**Why Mamba for 3D Pose?**
- ✅ Superior long-range modeling
- ✅ Linear complexity vs quadratic (Transformers)
- ✅ Fast inference (5× throughput)
- ✅ Better temporal modeling
- ✅ Scales to long videos

#### Diffusion Models (Probabilistic 3D Pose)

**HDPose** (2024) - Hierarchical Diffusion
- Post-hierarchical diffusion with conditioning
- Iterative denoising from noisy 3D pose
- No adversarial training (stable)
- Multiple plausible hypotheses
- 📄 [Paper](https://www.mdpi.com/1424-8220/24/3/829)

**FinePOSE** (May 2024)
- Fine-grained prompt-driven denoiser
- Part-aware prompt learning
- Diffusion-based generation
- 📄 [Paper](https://arxiv.org/abs/2405.05216)

**DiffuPose** (2024)
- Denoising Diffusion Probabilistic Model
- Handles depth ambiguity
- Generative modeling approach
- 📄 [arXiv](https://www.researchgate.net/publication/376499642)

**Key Innovation:** Diffusion models generate multiple hypotheses instead of single prediction, naturally handling depth ambiguity and occlusion.

---

### 📚 Monocular 3D Pose Methods

#### 2D-to-3D Lifting

**Classic Approach:**
```
RGB Image → 2D Pose Detector → 2D Keypoints → 3D Lifting Network → 3D Pose
```

| Method | Year | MPJPE (H3.6M) | Key Innovation | Code |
|--------|------|---------------|----------------|------|
| **Martinez Baseline** | 2017 | 37.7mm | Simple FC residual nets | [GitHub](https://github.com/una-dinosauria/3d-pose-baseline) |
| **SemGCN** | 2019 | 35.2mm | Semantic graph convolutions | [GitHub](https://github.com/garyzhao/SemGCN) |
| **MotionAGFormer** | 2024 | 33.4mm | Transformer-GCN hybrid | - |

**Advantages:**
- ✅ Leverage strong 2D detectors
- ✅ Modular design
- ✅ Can train separately
- ✅ Better generalization

**Martinez Method** (ICCV 2017) 🌟
- Foundational 2D→3D lifting baseline
- Simple feedforward network
- Still competitive in 2024
- Used in many recent works

#### End-to-End 3D Prediction

Direct regression from RGB to 3D pose without explicit 2D detection.

**Advantages:**
- ✅ No error accumulation
- ✅ Joint optimization
- ✅ Faster inference

---

### 🎬 Video-Based Temporal 3D Pose

Exploiting temporal consistency across frames for smoother, more accurate 3D pose.

#### Temporal Convolutional Approaches

**VideoPose3D** (CVPR 2019) 🌟
- Dilated temporal convolutions
- Semi-supervised training
- 46.8mm MPJPE on Human3.6M
- Foundational work for video-based methods
- 📄 [Paper](https://arxiv.org/abs/1811.11742) | 💻 [Code](https://github.com/facebookresearch/VideoPose3D)

**Architecture:**
```
2D Keypoints Sequence → Temporal Conv (dilated) → 3D Pose Sequence
```

#### Transformer-Based Temporal Methods

**PoseFormer** (ICCV 2021)
- First pure transformer for 3D pose
- Spatial-temporal attention
- Models joint relations + temporal correlations
- No convolutions required
- 📄 [Paper](https://arxiv.org/abs/2103.10455)

**MHFormer** (CVPR 2022) - Multi-Hypothesis
- Multiple hypothesis generation
- Transformer-based
- Improves representational power
- Synthesizes diverse pose hypotheses
- 💻 [Code](https://github.com/Vegetebird/MHFormer)

**MixSTE** (CVPR 2022) - Spatial-Temporal Excellence
- Joints as tokens (temporal + spatial)
- Preserves sequence coherence
- Mixed spatial-temporal encoding
- 💻 [Code](https://github.com/JinluZhang1126/MixSTE)

**P-STMO** (ECCV 2022)
- Pre-trained Spatial-Temporal Many-to-One
- Strong baseline for comparisons
- Used in many 2024 benchmarks

**MotionBERT** (ICCV 2023)
- Dual-stream spatio-temporal transformer
- Long-range dependencies
- Pre-training on large datasets
- 💻 [Code](https://github.com/Walter0807/MotionBERT)

#### Latest Efficiency Improvements (2024)

**Hourglass Tokenizer (HoT)** - CVPR 2024
- Reduces FLOPs by 50% on MotionBERT
- Reduces FLOPs by 40% on MixSTE
- Minimal performance loss (<0.2%)
- Hierarchical token compression
- 💻 [Code](https://github.com/NationalGAILab/HoT)

**DASTFormer** (2024)
- 39.6mm MPJPE (Protocol 1)
- 33.4mm P-MPJPE (Protocol 2)
- 7.5% improvement over P-STMO
- Dynamic attention mechanisms

#### Performance Comparison (Human3.6M)

| Method | Year | MPJPE | PA-MPJPE | Approach |
|--------|------|-------|----------|----------|
| **DASTFormer** | 2024 | **39.6mm** | **33.4mm** | Transformer |
| **MotionBERT** | 2023 | 41.2mm | 35.8mm | Transformer |
| **MixSTE** | 2022 | 42.9mm | 36.1mm | Transformer |
| **MHFormer** | 2022 | 43.0mm | 36.4mm | Transformer |
| **PoseFormer** | 2021 | 44.3mm | 37.2mm | Transformer |
| **VideoPose3D** | 2019 | 46.8mm | 36.5mm | TCN |

---

### 🎥 Multi-View 3D Pose Estimation

Using multiple synchronized cameras for accurate 3D reconstruction.

#### Voxel-Based Methods

**VoxelPose** (ECCV 2020) 🌟
- Projects 2D heatmaps to 3D voxel space
- Coarse-to-fine refinement
- Handles occlusion naturally
- Multi-person capable

**VoxelKeypointFusion** (October 2024)
- Learning-free algorithmic approach
- Voxel-based vs line-based triangulation
- Multiple keypoints per ray
- Detects occluded keypoints better
- 📄 [Paper](https://arxiv.org/abs/2410.18723)

**3DSA** (ECCV 2024) - 3D Space Attention
- Attention mechanisms in voxel space
- SOTA on CMU Panoptic Studio
- Improves VoxelPose and Faster VoxelPose

#### Triangulation-Based Methods

**RapidPoseTriangulation** (2024)
- Learning-free triangulation
- Multi-person whole-body
- Millisecond inference
- Simple and effective
- 📄 [Paper](https://arxiv.org/abs/2503.21692)

**Classical Approach:**
```
Multi-view 2D Poses → Epipolar Geometry → Triangulation → 3D Pose
```

#### Latest Hybrid Approaches (2024)

**Multiple View Geometry Transformers** (CVPR 2024)
- Transformer-based multi-view fusion
- End-to-end learning
- Superior to VoxelPose
- Reduces quantization error

**Comparison:**
- **Voxel-based**: Better occlusion handling, end-to-end trainable
- **Triangulation**: Faster, learning-free, interpretable
- **Hybrid**: Best accuracy, combines both approaches

---

### 🎨 3D Human Mesh Recovery (SMPL/SMPL-X)

Reconstructing full 3D human body mesh (not just keypoints).

#### SMPL Parametric Model

**SMPL** = Skinned Multi-Person Linear model
- **Vertices**: 6,890 vertices
- **Faces**: 13,776 faces
- **Parameters**:
  - β (10): Shape parameters
  - θ (72): Pose parameters (24 joints × 3 rotation)

**SMPL-X** = Extended SMPL
- Adds hands (MANO)
- Adds face expression
- Whole-body reconstruction

#### Latest SMPL Methods (2024-2025)

**ADHMR** (ICML 2025) 🔥
- Aligning Diffusion-based HMR
- Direct Preference Optimization
- Latest SOTA approach
- 💻 [Code](https://github.com/SMPLCap/ADHMR)

**Multi-HMR** (ECCV 2024)
- Multi-person whole-body in single shot
- SMPL-X predictions
- Hands + face + body
- 3D location in camera coordinates
- 📄 [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03364.pdf)

**CLIFF** (2024)
- Carrying Location Information in Full Frames
- Integrates spatial context
- Compatible with all HMR frameworks

**SMPLer-X** (NeurIPS 2023)
- Scaling up expressive pose
- Whole-body estimation
- Large-scale training

**RoboSMPLX** (NeurIPS 2023)
- Enhanced robustness
- Whole-body pose
- Handles difficult cases

#### Applications of SMPL
- 🎮 **Gaming**: Avatar creation
- 🎬 **VFX**: Digital humans
- 👗 **Fashion**: Virtual try-on
- 🏃 **Sports**: Biomechanics analysis
- 🏥 **Healthcare**: Gait analysis

---

### 📊 Major 3D Pose Datasets & Benchmarks

| Dataset | Year | Type | Frames | Subjects | Actions | Environment | MPJPE Baseline |
|---------|------|------|--------|----------|---------|-------------|----------------|
| **Human3.6M** | 2014 | Lab | 3.6M | 9 | 15 | Indoor, 4 cams | ~40mm |
| **3DPW** | 2018 | Wild | 51K | 7 | 47 | Outdoor, in-the-wild | ~47mm |
| **MPI-INF-3DHP** | 2017 | Mixed | 1.3M | 8 | 8 | Indoor + Outdoor, 14 cams | ~50mm |

**Human3.6M** 🌟
- Most popular benchmark
- Laboratory setting, high quality
- Protocols: P1 (MPJPE), P2 (PA-MPJPE)
- Standard for method comparison

**3DPW** 🌟
- In-the-wild outdoor scenes
- IMU sensors + video
- Real-world performance evaluation
- Challenging: occlusion, lighting, motion

**MPI-INF-3DHP**
- Both indoor & outdoor
- Green screen + studio backgrounds
- Multi-view (14 cameras)
- Markerless MoCap system

#### Recent Benchmark Datasets (2023-2024)

**H3WB** (ICCV 2023)
- Human3.6M 3D WholeBody
- 133 keypoints (body + hands + face)
- Extension of Human3.6M
- 💻 [GitHub](https://github.com/wholebody3d/wholebody3d)

**FreeMan** (CVPR 2024)
- Real-world conditions benchmark
- Diverse scenarios
- Addresses dataset bias

**AthletePose3D** (2024)
- Athletic movements
- Kinematic validation
- Sports-specific

---

### 📏 Evaluation Metrics

**MPJPE** (Mean Per Joint Position Error)
```python
MPJPE = mean(||pred_joints - gt_joints||₂)
```
- Unit: millimeters
- Protocol 1 on Human3.6M
- Direct 3D distance

**PA-MPJPE** (Procrustes Aligned MPJPE)
```python
PA-MPJPE = MPJPE after Procrustes alignment
```
- Removes global rotation, scale, translation
- Protocol 2 on Human3.6M
- Focuses on pose structure

**P-MPJPE** (Per-joint MPJPE)
- Individual joint errors
- Identifies weak joints

**N-MPJPE** (Normalized MPJPE)
- Normalized by torso size
- Scale-invariant evaluation

---

### 💻 Quick Start Example

#### VideoPose3D Inference

```python
import torch
from common.model import TemporalModel

# Load model
model = TemporalModel(
    num_joints_in=17,
    in_features=2,
    num_joints_out=17,
    filter_widths=[3,3,3,3,3],
    causal=False,
    dropout=0.25,
    channels=1024
)

checkpoint = torch.load('pretrained_h36m_detectron_coco.bin')
model.load_state_dict(checkpoint['model_pos'])
model.eval()

# Input: 2D keypoints sequence [T, 17, 2]
# Output: 3D pose sequence [T, 17, 3]
with torch.no_grad():
    predicted_3d = model(keypoints_2d)
```

#### MotionBERT Inference

```python
from lib.model.DSTformer import DSTformer

# Load model
model = DSTformer(
    dim_in=3,
    dim_out=3,
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=2
)

# Input: [B, T, J, C] - Batch, Time, Joints, Channels
# Output: [B, T, J, 3] - 3D coordinates
output_3d = model(input_2d)
```

---

### 🔮 Latest Research Trends (2024-2025)

1. **State-Space Models (Mamba)** 🔥
   - Linear complexity
   - Superior to Transformers for long sequences
   - SasMamba, PoseMamba leading methods

2. **Diffusion Models** 🔥
   - Probabilistic 3D pose
   - Multiple hypotheses
   - Better uncertainty modeling

3. **Foundation Models**
   - Large-scale pre-training
   - Cross-dataset generalization
   - Few-shot adaptation

4. **Neural Radiance Fields (NeRF)**
   - 3D scene representation
   - Novel view synthesis
   - Implicit 3D modeling

5. **Self-Supervised Learning**
   - Monocular depth estimation
   - Unlabeled video exploitation
   - Reduced annotation cost

6. **Real-Time Optimization**
   - Efficient tokenization (HoT)
   - Model compression
   - Edge deployment

7. **Multi-Modal Fusion**
   - Vision + IMU sensors
   - RGB-D integration
   - Audio-visual cues

---

### 🎯 Applications

- **🥽 AR/VR**: Full-body tracking for metaverse
- **🎬 Motion Capture**: Film & gaming animation
- **⚽ Sports Analytics**: Biomechanics, performance analysis
- **🏥 Healthcare**: Gait analysis, rehabilitation monitoring
- **🚗 Autonomous Driving**: Pedestrian pose understanding
- **🤖 Robotics**: Human-robot interaction, imitation learning
- **🎮 Gaming**: Real-time character control
- **👮 Surveillance**: Behavior analysis, anomaly detection

---

### 📚 Key Resources

**Benchmarks & Leaderboards:**
- [Papers With Code - 3D HPE](https://paperswithcode.com/task/3d-human-pose-estimation)
- [Human3.6M Leaderboard](http://vision.imar.ro/human3.6m/ranking.php)

**Repositories:**
- [Awesome 3D Human Pose](https://github.com/3dhumanbody/awesome-3d-human)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [HoT (Hourglass Tokenizer)](https://github.com/NationalGAILab/HoT)

**Latest Surveys:**
- [Monocular 3D HPE Survey (April 2025)](https://www.mdpi.com/1424-8220/25/8/2409)
- [Deep 3D HPE Survey (2024)](https://link.springer.com/article/10.1007/s10462-024-11019-3)

---

## ✋ Hand Pose Estimation

> 📖 **[Complete Hand & Finger Pose Guide](HAND_POSE.md)** - Ultra-comprehensive documentation covering all aspects of hand pose estimation

Detecting and tracking hand joints and finger positions - one of the most challenging problems in computer vision with 27 DOF and severe self-occlusion.

### 🎯 Key Areas

<table>
<tr>
<td width="25%">

**2D/3D Pose**
- 21 keypoint detection
- Monocular RGB methods
- RGB-D approaches
- Real-time tracking

</td>
<td width="25%">

**3D Mesh Reconstruction**
- MANO parametric model
- HaMeR (2024 SOTA)
- MeshGraphormer
- 778 vertices output

</td>
<td width="25%">

**Hand-Object Interaction**
- Grasping analysis
- Contact modeling
- Joint reconstruction
- Physics-based methods

</td>
<td width="25%">

**Applications**
- Sign language (98%+ accuracy)
- VR/AR interaction
- Gesture recognition
- Robot manipulation

</td>
</tr>
</table>

### 🔥 State-of-the-Art Methods (2024-2025)

#### 3D Hand Mesh Reconstruction

| Method | Year | Type | PA-MPJPE | Key Innovation | Code |
|--------|------|------|----------|----------------|------|
| **HaMeR** | 2024 | Parametric | 5.6mm | Transformer-based, SOTA accuracy | [GitHub](https://github.com/geopavlakos/hamer) |
| **Hamba** | 2024 | Parametric | 5.2mm | Mamba architecture, bi-scanning | [arXiv](https://arxiv.org/abs/2407.09646) |
| **MaskHand** | 2024 | Parametric | 5.1mm | Masked modeling, 7.5% improvement | [arXiv](https://arxiv.org/abs/2412.13393) |
| **MeshGraphormer** | 2021 | Non-parametric | 6.0mm | Graph transformer | [GitHub](https://github.com/microsoft/MeshGraphormer) |

#### Real-Time Hand Tracking

**MediaPipe Hands** 🌟
- ✅ 21 3D landmarks @ 30+ FPS
- ✅ Multi-hand support (up to 2 hands)
- ✅ Cross-platform (mobile, web, desktop)
- ✅ Palm detection + landmark model
- 📄 [Paper](https://arxiv.org/abs/2006.10214) | 💻 [Code](https://github.com/google/mediapipe)

**Performance:**
- Accuracy: 95%+ on palm detection
- Latency: 33ms on Pixel 3
- Landmarks: <5% error relative to palm size

### 🤝 Hand-Object Interaction (2024-2025)

Recent breakthroughs in understanding how hands interact with objects:

| Method | Venue | Innovation | Application |
|--------|-------|------------|-------------|
| **HOLD** | CVPR 2024 | First template-free HOI from video | Articulated objects |
| **HOIC** | SIGGRAPH 2024 | Physics-based RL reconstruction | RGBD manipulation |
| **DiffH2O** | SIGGRAPH Asia 2024 | Text-to-interaction generation | Dexterous grasping |
| **ManiVideo** | CVPR 2025 | Hand-object manipulation video | Generalizable grasping |
| **HOISDF** | CVPR 2024 | Global SDF constraints | SOTA on DexYCB |

### 👐 Two-Hand Interaction

**InterHand2.6M Dataset** - First large-scale dataset for interacting hands
- 📊 2.6M annotated frames
- 🤝 Single + interacting hands
- 📐 3D joint locations (42 keypoints)

**Recent Methods:**
- **HandFI** (2024): Multi-level feature fusion
- **VM-BHINet** (2025): Vision Mamba for bimanual hands
- **InterHandGen** (2024): Diffusion-based generation

### 🥽 Egocentric Hand Pose

Critical for AR/VR applications with unique challenges:
- Close-range perspective distortion
- Partial visibility
- Motion blur
- Limited field of view

**Recent Solutions:**
- Multi-view egocentric tracking (Meta Quest 3, Apple Vision Pro)
- ECCV 2024 Challenge winner: 13.92mm MPJPE
- EgoWorld (2025): Exo-to-ego view translation

### 🤟 Sign Language Recognition

**Real-Time ASL (2025):** 98.2% accuracy using YOLOv11 + MediaPipe
- Real-time inference on standard webcam
- Handles visually similar gestures (A/T, M/N)
- mAP@0.5: 98.2%

**Applications:**
- American Sign Language (ASL) alphabet
- Continuous sign language translation
- Fingerspelling recognition (71.7% SOTA)

### 📊 Major Datasets

| Dataset | Year | Type | Samples | Features | Links |
|---------|------|------|---------|----------|-------|
| **FreiHAND** | 2019 | RGB | 134K | 21 joints + mesh | [Website](https://lmb.informatik.uni-freiburg.de/projects/freihand/) |
| **RHD** | 2017 | RGB | 44K | Synthetic hands | [Website](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) |
| **InterHand2.6M** | 2020 | RGB | 2.6M | Two-hand interaction | [Website](https://mks0601.github.io/InterHand2.6M/) |
| **HO-3D** | 2020 | RGB-D | 77K | Hand + object | [GitHub](https://github.com/shreyashampali/ho3d) |
| **DexYCB** | 2021 | RGB-D | 582K | Grasping | [Website](https://dex-ycb.github.io/) |
| **ContactPose** | 2020 | RGB-D | 2.9K | Contact annotations | [Website](https://contactpose.cc.gatech.edu/) |

### 🛠️ Production Tools

**Frameworks:**
- **MediaPipe**: Real-time, cross-platform
- **MMPose**: 200+ models, research-focused
- **HaMeR**: 3D mesh reconstruction
- **Detectron2**: Keypoint R-CNN

**Specialized:**
- **MANO/Manopth**: Parametric hand model
- **PyTorch3D**: 3D deep learning
- **Open3D**: Point cloud processing

### 💻 Quick Start

```python
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Process frame
image = cv2.imread('hand.jpg')
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Get 21 keypoints per hand
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Access individual joints: wrist (0), thumb_tip (4), index_tip (8), etc.
        wrist = hand_landmarks.landmark[0]
        print(f"Wrist: x={wrist.x}, y={wrist.y}, z={wrist.z}")
```

### 🎯 Applications & Use Cases

- **🥽 AR/VR**: Natural hand interaction in virtual environments
- **🤖 Robotics**: Teaching by demonstration, teleoperation
- **🏥 Healthcare**: Rehabilitation monitoring, surgical training
- **🎮 Gaming**: Gesture-based controls, motion capture
- **♿ Accessibility**: Sign language translation, assistive tech
- **🚗 Automotive**: Touchless infotainment controls

### 📏 Evaluation Metrics

- **MPJPE**: Mean Per Joint Position Error (mm)
- **PA-MPJPE**: Procrustes Aligned MPJPE
- **AUC**: Area Under PCK Curve (0-50mm)
- **PCK**: Percentage of Correct Keypoints
- **F@Xmm**: Fraction of frames under X mm error

### 🔮 Latest Trends (2024-2025)

1. **Transformer dominance**: ViT-based architectures outperform CNNs
2. **Foundation models**: Large-scale pre-training for generalization
3. **Generative approaches**: Diffusion models for hand synthesis
4. **Physics integration**: Biomechanical constraints, contact modeling
5. **Multimodal fusion**: Vision + IMU + tactile sensing

---

> 📚 **For comprehensive coverage**, see our [Complete Hand & Finger Pose Estimation Guide](HAND_POSE.md) with:
> - Detailed method explanations
> - Complete code examples
> - Dataset comparisons
> - Benchmark results
> - Implementation tutorials

---

## 😊 Face & Facial Landmark Detection

Detecting facial keypoints for face alignment and analysis.

### 🔧 Popular Libraries & Tools

**Dlib Face Alignment**
- 68-point facial landmark detector
- Based on ensemble of regression trees
- Trained on iBUG 300-W dataset
- Fast and accurate for frontal faces
- 📄 [Paper](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf) | [Code](https://github.com/davisking/dlib)

**MediaPipe Face Mesh**
- 468 3D facial landmarks
- Real-time on mobile devices
- Handles various poses and expressions
- [Code](https://github.com/google/mediapipe)

**OpenFace**
- 2D and 3D facial landmark detection
- Facial action unit recognition
- Head pose estimation
- [GitHub](https://github.com/TadasBaltrusaitis/OpenFace)

### Landmark Configurations
- 🔸 **5 points**: Eyes, nose, mouth corners (alignment)
- 🔸 **68 points**: dlib standard (iBUG 300-W)
- 🔸 **98 points**: WFLW dataset
- 🔸 **468 points**: MediaPipe Face Mesh

### Key Challenges
- Occlusions (masks, hands, hair)
- Extreme head poses
- Lighting variations
- Expression changes

---

## 👤 Whole-Body Pose Estimation

Unified estimation of body, hands, and face keypoints.

### 🎯 Methods

**AlphaPose** 🌟
- Regional multi-person whole-body pose
- Real-time tracking
- Body + hands + face + feet
- 📄 [Paper](https://ieeexplore.ieee.org/document/9954214) | [Code](https://github.com/MVIG-SJTU/AlphaPose)

**Recent Advances (2024-2025)**

| Method | Year | Keypoints | Features |
|--------|------|-----------|----------|
| **EE-YOLOv8** | 2025 | 133 | EMRF + EFPN architecture |
| **ZoomNAS** | 2022 | 133 | Neural architecture search |
| **DWPose** | 2023 | 133 | Distilled whole-body pose |

### Keypoint Breakdown
- 👤 **Body**: 17 points (COCO format)
- ✋ **Hands**: 21 points each (42 total)
- 😊 **Face**: 68-70 points
- 🦶 **Feet**: 6 points
- **Total**: ~133 keypoints

---

## 👥 Multi-Person Pose Estimation

Detecting and estimating poses for multiple people in crowded scenes.

### Approaches

#### Top-Down Approach
1. Detect person bounding boxes (object detector)
2. Estimate pose for each person independently
- ✅ High accuracy
- ❌ Speed decreases with more people
- 🔧 **Methods**: Faster R-CNN + pose, YOLO + pose

#### Bottom-Up Approach
1. Detect all keypoints in image
2. Group keypoints into individuals
- ✅ Speed independent of people count
- ❌ Lower accuracy in crowded scenes
- 🔧 **Methods**: OpenPose, HigherHRNet

#### Hybrid Approach (Best of Both) 🔥

**YOLO-Pose**
- Joint detection + pose estimation
- Single forward pass
- COCO val: 90.2% AP50
- Real-time performance
- 📄 [Paper](https://arxiv.org/abs/2204.06806) | [Code](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

**YOLO11-Pose** (2024)
- Latest YOLO variant for pose
- Anchor-free, single-stage
- Optimized for speed + accuracy
- [Docs](https://www.ultralytics.com/)

---

## ⚡ Real-Time & Lightweight Models

Models optimized for edge devices, mobile, and real-time applications.

### 📱 Mobile-Optimized Models

| Model | FPS | Size | Target | Links |
|-------|-----|------|--------|-------|
| **MoveNet** | >50 | <10MB | Mobile, edge | [TF Hub](https://tfhub.dev/s?q=movenet) |
| **PoseNet** | 30+ | 13MB | Browser, mobile | [TensorFlow](https://www.tensorflow.org/lite/examples/pose_estimation/overview) |
| **Lite-HRNet** | 25+ | <10MB | Mobile | [GitHub](https://github.com/HRNet/Lite-HRNet) |
| **MobilePose** | 30+ | 5MB | Mobile | - |
| **PocketPose** | 40+ | <5MB | Edge devices | [PyPI](https://pypi.org/project/pocketpose/) |

### 🚀 Deployment Frameworks

**TensorFlow Lite**
- Convert models to .tflite format
- Quantization support (INT8, FP16)
- Optimized for mobile/edge
- [Guide](https://www.tensorflow.org/lite/examples/pose_estimation/overview)

**ONNX Runtime**
- Cross-platform inference
- Hardware acceleration
- Supports multiple backends
- [Docs](https://onnxruntime.ai/)

**MediaPipe**
- Ready-to-use solutions
- Cross-platform (iOS, Android, Web)
- Optimized pipelines
- [Solutions](https://developers.google.com/mediapipe)

### Performance Optimization Techniques
- 🔧 Model quantization (INT8, FP16)
- 🔧 Knowledge distillation
- 🔧 Neural architecture search (NAS)
- 🔧 Pruning and compression
- 🔧 Hardware-aware design

---

## 📊 Datasets

Comprehensive list of pose estimation datasets with benchmarks.

### 2D Pose Datasets

| Dataset | Year | Images | People | Keypoints | Annotations | Links |
|---------|------|--------|--------|-----------|-------------|-------|
| **COCO** | 2014 | 250K+ | 250K+ | 17 | 2D body | [Website](https://cocodataset.org/) |
| **MPII** | 2014 | 25K | 40K | 16 | 2D body | [Website](http://human-pose.mpi-inf.mpg.de/) |
| **AI Challenger** | 2017 | 300K+ | 700K+ | 14 | 2D body | - |
| **CrowdPose** | 2019 | 20K | 80K | 14 | Crowded scenes | [GitHub](https://github.com/Jeff-sjtu/CrowdPose) |
| **OCHuman** | 2019 | 5K | 13K | 17 | Occlusions | [GitHub](https://github.com/liruilong940607/OCHumanApi) |

### 3D Pose Datasets

| Dataset | Year | Type | Subjects | Keypoints | Environment | Links |
|---------|------|------|----------|-----------|-------------|-------|
| **Human3.6M** | 2014 | 3D | 11 | 32 (reduced to 17) | Indoor | [Website](http://vision.imar.ro/human3.6m/) |
| **MPI-INF-3DHP** | 2017 | 3D | 8 | 17 | Indoor + outdoor | [Website](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) |
| **3DPW** | 2018 | 3D + mesh | 7 | 18 | Outdoor | [Website](https://virtualhumans.mpi-inf.mpg.de/3DPW/) |
| **H3WB** | 2023 | 3D whole-body | - | 133 | Extension of H3.6M | [GitHub](https://github.com/wholebody3d/wholebody3d) |

### Hand Pose Datasets

| Dataset | Year | Type | Samples | Keypoints | Links |
|---------|------|------|---------|-----------|-------|
| **FreiHAND** | 2019 | 3D | 130K | 21 | [GitHub](https://github.com/lmb-freiburg/freihand) |
| **InterHand2.6M** | 2020 | 3D | 2.6M | 21 per hand | [GitHub](https://github.com/facebookresearch/InterHand2.6M) |
| **RHD** | 2017 | 3D | 44K | 21 | - |
| **STB** | 2017 | 3D | 18K | 21 | - |

### Face & Whole-Body Datasets

| Dataset | Year | Type | Landmarks | Notes |
|---------|------|------|-----------|-------|
| **300W** | 2013 | Face | 68 | iBUG standard |
| **WFLW** | 2018 | Face | 98 | Large pose variations |
| **AFLW** | 2011 | Face | 21 | In-the-wild |
| **COCO-WholeBody** | 2020 | Whole-body | 133 | Body + hands + face |
| **Halpe** | 2020 | Whole-body | 136 | Extended keypoints |

### Specialized Datasets

- **AthletePose3D** (2024): Athletic movements, kinematic validation
- **SURREAL**: Synthetic humans, ground truth
- **Human4D**: 4D human scans with motion
- **PoseTrack**: Video pose estimation, tracking

---

## 📏 Evaluation Metrics

Understanding how pose estimation models are evaluated.

### 📊 Common Metrics

#### PCK (Percentage of Correct Keypoints)
- Keypoint is correct if within threshold of ground truth
- **PCK@0.5**: 50% of reference distance
- **PCKh@0.5**: 50% of head segment length (MPII)
- Range: 0-100% (higher is better)

```
PCK = (# correct keypoints) / (# total keypoints) × 100%
```

#### AP (Average Precision) / mAP
- Based on Object Keypoint Similarity (OKS)
- Primary metric for COCO dataset
- Computed at multiple OKS thresholds
- **AP@0.5, AP@0.75, AP@[0.5:0.95]**

#### OKS (Object Keypoint Similarity)
- Similar to IoU for object detection
- Normalized by object scale
- Accounts for keypoint-specific uncertainty

```
OKS = Σᵢ exp(-dᵢ²/2s²kᵢ²) δ(vᵢ>0) / Σᵢ δ(vᵢ>0)
```
- dᵢ: Euclidean distance
- s: object scale
- kᵢ: keypoint constant

#### MPJPE (Mean Per Joint Position Error)
- For 3D pose estimation
- Average Euclidean distance in mm
- **PA-MPJPE**: After Procrustes alignment
- Lower is better

```
MPJPE = (1/N) Σᵢ ||pᵢ - p̂ᵢ||₂
```

### Dataset-Specific Metrics

| Dataset | Primary Metric | Secondary Metrics |
|---------|----------------|-------------------|
| COCO | AP (mAP) | AP@0.5, AP@0.75, AR |
| MPII | PCKh@0.5 | PCKh@0.1, AUC |
| Human3.6M | MPJPE | PA-MPJPE, P-MPJPE |
| 3DPW | MPJPE | PA-MPJPE |

---

## 🛠️ Tools & Libraries

Production-ready tools for pose estimation.

### 🔥 Major Frameworks

#### **MMPose**
- Part of OpenMMLab ecosystem
- 200+ pre-trained models
- Supports 2D/3D, single/multi-person
- [GitHub](https://github.com/open-mmlab/mmpose) | [Docs](https://mmpose.readthedocs.io/)

```python
from mmpose.apis import init_model, inference_topdown
model = init_model(config, checkpoint)
results = inference_topdown(model, image)
```

#### **MediaPipe**
- Google's ML solutions
- Ready-to-use pose, hand, face
- Cross-platform (Python, JS, C++)
- [Website](https://developers.google.com/mediapipe)

```python
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
results = pose.process(image)
```

#### **OpenPose**
- First real-time multi-person
- Body + hands + face
- C++ with Python API
- [GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

#### **AlphaPose**
- Multi-person pose tracking
- Whole-body support
- Video processing
- [GitHub](https://github.com/MVIG-SJTU/AlphaPose)

#### **Detectron2**
- Facebook AI Research
- Keypoint R-CNN implementation
- PyTorch-based
- [GitHub](https://github.com/facebookresearch/detectron2)

#### **Ultralytics YOLO**
- YOLO11-Pose (latest)
- Real-time inference
- Easy deployment
- [GitHub](https://github.com/ultralytics/ultralytics)

```python
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
results = model('image.jpg')
```

### 🧰 Specialized Libraries

| Library | Purpose | Language | Links |
|---------|---------|----------|-------|
| **PocketPose** | Mobile/edge deployment | Python | [PyPI](https://pypi.org/project/pocketpose/) |
| **VitePose** | Fast inference | Python | - |
| **tf-pose-estimation** | TensorFlow implementation | Python | [GitHub](https://github.com/ildoonet/tf-pose-estimation) |
| **PyTorch-Pose** | PyTorch models | Python | [GitHub](https://github.com/bearpaw/pytorch-pose) |
| **PoseEstimationForMobile** | Mobile (iOS/Android) | Swift/Java | [GitHub](https://github.com/edvardHua/PoseEstimationForMobile) |

### 📦 Conversion & Deployment

- **ONNX**: Model interoperability
- **TensorRT**: NVIDIA GPU acceleration
- **OpenVINO**: Intel hardware optimization
- **CoreML**: Apple devices
- **TFLite**: Mobile/embedded

---

## 🎯 Applications

Real-world use cases of pose estimation.

### 🏃 Sports & Fitness
- Form analysis and correction
- Rep counting and tracking
- Performance analytics
- Injury prevention
- Virtual coaching

### 🎮 Gaming & Entertainment
- Motion capture for games/movies
- Virtual reality interaction
- Augmented reality filters
- Gesture-based controls
- Avatar animation

### 🏥 Healthcare & Rehabilitation
- Gait analysis
- Physical therapy monitoring
- Elderly fall detection
- Movement disorder assessment
- Ergonomic analysis

### 🤖 Human-Computer Interaction
- Touchless interfaces
- Sign language recognition
- Smart home controls
- Security and surveillance
- Driver monitoring

### 🎬 Content Creation
- Video editing and effects
- Animation retargeting
- Virtual try-on
- Social media filters
- Live performance capture

### 🏭 Industry & Research
- Worker safety monitoring
- Assembly line optimization
- Biomechanics research
- Crowd behavior analysis
- Human-robot collaboration

---

## 📚 Resources

### 📖 Learning Resources

**Tutorials & Guides**
- [Real-Time Pose Estimation Guide](https://viso.ai/deep-learning/pose-estimation-ultimate-overview/)
- [Human Pose Estimation Deep Learning Guide](https://www.v7labs.com/blog/human-pose-estimation-guide)
- [MediaPipe Pose Tutorial](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

**Courses & Workshops**
- Computer Vision courses (Stanford CS231n, etc.)
- Deep Learning Specialization (Coursera)
- PyTorch/TensorFlow tutorials

### 🌐 Benchmarks & Leaderboards

- [Papers With Code - Pose Estimation](https://paperswithcode.com/task/pose-estimation)
- [Papers With Code - 3D Pose](https://paperswithcode.com/task/3d-human-pose-estimation)
- [COCO Keypoint Challenge](https://cocodataset.org/#keypoints-leaderboard)

### 🔬 Research Groups & Labs

- **CMU Perceptual Computing Lab**: OpenPose creators
- **Facebook AI Research (FAIR)**: Detectron2, 3DPW
- **Google Research**: MediaPipe
- **Microsoft Research**: HRNet, various datasets
- **Shanghai Jiao Tong University**: AlphaPose

### 📰 Conferences & Journals

**Top Venues:**
- CVPR (Computer Vision and Pattern Recognition)
- ICCV (International Conference on Computer Vision)
- ECCV (European Conference on Computer Vision)
- NeurIPS (Neural Information Processing Systems)
- TPAMI (IEEE Transactions on Pattern Analysis and Machine Intelligence)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Add papers with publication year and venue
- Include links to paper, code, and project page
- Maintain alphabetical or chronological order
- Update table of contents if adding new sections
- Follow existing formatting style

---

## 📄 License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

If you find this repository useful, please consider giving it a star ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=umitkacar/Pose-Estimation&type=Date)](https://star-history.com/#umitkacar/Pose-Estimation&Date)

---

## 🙏 Acknowledgments

Thanks to all the researchers and developers who have contributed to the field of human pose estimation. This repository stands on the shoulders of giants.

Special thanks to:
- OpenMMLab for MMPose
- Google for MediaPipe
- CMU for OpenPose
- Facebook Research for Detectron2
- Ultralytics for YOLO
- All dataset creators and maintainers

---

<div align="center">

### 📧 Contact & Questions

For questions or suggestions, please open an issue or contact the maintainer.

**Last Updated:** January 2025

**Maintained with ❤️ by the Computer Vision Community**

</div>

---

## 🔖 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{awesome-pose-estimation,
  author = {Pose Estimation Community},
  title = {Awesome Human Pose Estimation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/umitkacar/Pose-Estimation}
}
```

---

<div align="center">
  <sub>Built with 🔥 by researchers, for researchers</sub>
</div>
