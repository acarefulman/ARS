# ARS: Attention-based Residual Selection for Unified Hierarchical Aggregation in Multimodal Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ARS** is a research project that leverages **attention-based residual selection** and **unified hierarchical aggregation** to improve multimodal recommendation systems.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Requirements & Dependencies](#requirements--dependencies)
- [Datasets](#datasets)
- [Experimental Pipeline](#experimental-pipeline)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation](#3-evaluation)
- [Results](#results)
- [Reproduction Guide](#reproduction-guide)
- [Citation](#citation)

---

## Overview

**ARS (Attention-based Residual Selection)** is designed for multimodal recommendation scenarios. The core idea is to **adaptively select** salient features from different modalities (e.g., text, images) via an attention mechanism, and then **hierarchically aggregate** them with residual connections. This approach enhances both the accuracy and robustness of the recommendation system.

---

## Requirements & Dependencies

The project is built with Python 3.8+ and relies on the following key libraries:

```bash
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
