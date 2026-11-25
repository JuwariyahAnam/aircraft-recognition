# Project guide → implementation map

How each part of the [project guide](Aircraft_Recognition_Project_Guide.pdf) is implemented in this repository.
The [full documentation](AERO-ID_Project_Documentation.pdf) covers everything in more detail.

## Objectives (guide section 7)

| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Router that identifies image, video or audio | Done. Checks file contents (magic bytes) and extension; rejects non-media | `aircraft_recog/router.py` |
| 2 | Prepare and synthesize about 5,000 clean samples | Done. 3,500 images + 1,000 spectrograms + 500 video frames | `data_prep.py`, `convert.py`, `synthesis.py` |
| 3 | Segment the aircraft with U-Net | Done. ResNet-34 encoder, Dice + BCE, IoU / Dice reported | `masks.py`, `segmentation.py` |
| 4 | Classify the type with a CNN | Done. ResNet-18 image and audio CNNs, plus exact-model CNN | `classifier.py`, `fine_grained.py` |
| 5 | Optimize with PSO and measure the improvement | Done. PSO vs GA vs Grey Wolf vs random, same 30-run budget | `optimizers.py`, `hpo.py` |
| 6 | Clustering + pseudo-labelling for unlabelled data | Done. K-means / DBSCAN, t-SNE, pseudo-labels above 0.90 | `semi_supervised.py` |
| 7 | Explain every prediction with Grad-CAM | Done. Grad-CAM, LIME, CAM in the browser | `explain.py`, `static/web_infer.js` |
| 8 | Demo: upload a file, see the result | Done. Gradio app + GitHub Pages website | `app.py`, `index.html` |

## Requirement checklist (guide section 8)

| What was asked | How it is covered | Where |
|---|---|---|
| CV, ML and DL concepts | CV: preprocessing, GrabCut, frames. ML: SVM, K-means, DBSCAN, PSO. DL: U-Net, CNNs | `masks.py`, `baseline_svm.py`, `classifier.py` |
| Image, video or audio input | Router sends each file down the right path | `router.py`, `pipeline.py` |
| Agent distinguishes the inputs | Contents win over extension; non-aircraft below 55% confidence flagged | `router.py`, `pipeline.py` |
| About 5,000 samples | 3,500 / 1,000 / 500 | `data_prep.py`, `data/manifest.csv` |
| Labelled and unlabelled | 60% / 40% of training labels | `semi_supervised.py` |
| Clean, structured input | Resize, denoise, normalise, fixed layout, group-aware 70/15/15 split | `data_prep.py`, `datasets.py` |
| Synthesize small → large | Augmentation, SpecAugment, optional GAN, seed-growth experiment | `synthesis.py` |
| Processing and segmentation | GrabCut masks + U-Net | `masks.py`, `segmentation.py` |
| Deep learning architecture | U-Net + ResNet-18 transfer learning | `segmentation.py`, `classifier.py` |
| Nature-inspired algorithm | PSO, Genetic Algorithm, Grey Wolf, written from scratch | `optimizers.py` |
| Optimization | Before / after table, convergence curve | `hpo.py` |
| Classify and cluster | 4 classes + exact model; K-means clusters | `classifier.py`, `fine_grained.py`, `semi_supervised.py` |
| Explainable AI | Grad-CAM + LIME | `explain.py` |
| Real application | Aerospace; retrainable for medical or bio images | whole project |

## Training settings (guide section 13)

| Setting | U-Net | CNN |
|---|---|---|
| Input size | 256 × 256 | 224 × 224 |
| Pretrained | ResNet-34 encoder, ImageNet | ResNet-18, ImageNet |
| Loss | Dice + BCE | Cross-entropy |
| Optimizer | Adam | Adam |
| Learning rate | 0.0001 | 0.001, then PSO (10⁻⁴ to 10⁻²) |
| Batch size | 16 | 32, then PSO (16 / 32 / 64) |
| Epochs | 25–30 | 15–20, early stopping |
| Metric | IoU, Dice | Accuracy, macro F1 |

PSO update: `v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`, `x = x + v`, with w = 0.7, c1 = c2 = 1.5 (`optimizers.py`).

## Evaluation (guide section 15)

| Part | Metric | Output file in `results/` |
|---|---|---|
| Segmentation | IoU, Dice | `unet_metrics.json`, `unet_samples.png` |
| Classification | Accuracy, precision, recall, F1 | `final_test_metrics.json`, `confusion_image.png`, `confusion_audio.png` |
| Optimization | Accuracy before / after PSO | `optimizer_comparison.csv`, `optimizer_convergence.png` |
| Semi-supervised | Accuracy with / without pseudo-labels | `semi_supervised.json`, `tsne_image.png`, `tsne_audio.png` |
| Explainability | Visual check | `gradcam_image.png`, `gradcam_audio.png`, `gradcam_vs_lime.png` |
| Results table | Image / video / audio accuracy per experiment | `results_table.csv` |

## Six-week plan (guide section 16)

| Week | Work | Notebook steps |
|---|---|---|
| 1 | Datasets, classes, class mapping | Step 1 |
| 2 | Conversion, cleaning, augmentation, splits, router | Steps 2–3 |
| 3 | Masks, U-Net | Steps 4–5 |
| 4 | CNN classifiers, baseline | Steps 6, 6b |
| 5 | PSO, other optimizers, clustering, pseudo-labels, SVM | Steps 7–8, extras |
| 6 | Grad-CAM, ablation, demo, website, report and slides | Steps 9–11 |

## What you submit (guide section 18)

| Deliverable | Where |
|---|---|
| Colab notebook organised by step | [`notebooks/Aircraft_Recognition_Colab.ipynb`](../notebooks/Aircraft_Recognition_Colab.ipynb) |
| Trained models and the 5,000-sample list | `models/`, `web_models/`, `data/manifest.csv` (created by training) |
| Results table, confusion matrix, PSO curve, t-SNE, heatmaps | `results/` (created by training) |
| Working demo app | `app.py` and the [live site](https://juwariyahanam.github.io/aircraft-recognition/) |
| Report and slides | Use the report structure in the documentation, section 20 |
