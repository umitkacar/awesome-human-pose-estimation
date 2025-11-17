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

Estimating 3D joint locations in world coordinates from images or videos.

### Monocular 3D Pose

Recovering 3D pose from single RGB images - the most challenging setting.

#### 🔬 Recent Advances (2024-2025)

**State-Space Models & Diffusion Models**
- Novel approaches incorporating SSMs and diffusion for 3D HPE
- Handle depth ambiguity through probabilistic modeling
- Generate diverse, plausible pose hypotheses

**Generative Models**
- GANs, VAEs, Diffusion Models for 3D pose
- Capture distribution of plausible poses
- Handle occlusion and depth ambiguity

#### 📚 Key Papers

| Method | Year | Venue | Approach | Code |
|--------|------|-------|----------|------|
| **VideoPose3D** | 2019 | CVPR | Temporal convolutions on 2D poses | [GitHub](https://github.com/facebookresearch/VideoPose3D) |
| **SimpleBaseline3D** | 2017 | ICCV | Lifting 2D to 3D | [GitHub](https://github.com/una-dinosauria/3d-pose-baseline) |
| **MotionBERT** | 2022 | ICCV | Pre-training for 3D pose | [GitHub](https://github.com/Walter0807/MotionBERT) |
| **MHFormer** | 2022 | CVPR | Multi-hypothesis transformer | [GitHub](https://github.com/Vegetebird/MHFormer) |
| **MixSTE** | 2022 | CVPR | Spatial-temporal transformer | [GitHub](https://github.com/JinluZhang1126/MixSTE) |

### Multi-View 3D Pose

Using multiple synchronized cameras for 3D reconstruction.

- **Triangulation-based**: Traditional multi-view geometry
- **Volume-based**: 3D voxel representations
- **Graph-based**: Multi-view fusion via graphs

### Video-Based 3D Pose

Leveraging temporal information from video sequences.

**Key Approaches:**
- 🎬 Temporal convolutional networks (TCNs)
- 🎬 Recurrent networks (LSTMs, GRUs)
- 🎬 Transformer-based temporal modeling
- 🎬 Hybrid approaches with IMU sensors

---

## ✋ Hand Pose Estimation

Detecting and tracking hand joints and finger positions.

### 🌟 Popular Methods & Frameworks

**MediaPipe Hands** 🔥
- On-device real-time hand tracking
- Detects 21 3D hand landmarks
- Palm detection + hand landmark model
- Runs on mobile, web, desktop
- 📄 [Paper](https://arxiv.org/abs/2006.10214) | [Code](https://github.com/google/mediapipe)

**Other Notable Methods:**

| Method | Year | Type | Key Features | Code |
|--------|------|------|--------------|------|
| **FreiHAND** | 2019 | 3D | Dataset + benchmark | [GitHub](https://github.com/lmb-freiburg/freihand) |
| **InterHand2.6M** | 2020 | 3D | Large-scale two-hand dataset | [GitHub](https://github.com/facebookresearch/InterHand2.6M) |
| **HandOccNet** | 2022 | 3D | Occlusion-aware hand pose | [GitHub](https://github.com/namepllet/HandOccNet) |
| **A2J** | 2019 | 3D | Anchor-based 3D hand pose | [GitHub](https://github.com/zhangboshen/A2J) |

### Applications
- 👆 Hand gesture recognition
- 🎮 VR/AR interaction
- 🤖 Human-computer interaction
- 🎵 Sign language recognition

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
