# CDVQA Dataset Documentation

## 1. Overview
The **CDVQA** (Change Detection-based Visual Question Answering) dataset is an aerial and satellite remote-sensing benchmark designed for visual question answering regarding bi-temporal surface changes. It is grounded on high-resolution optical imagery from the **SECOND** (SEmantic Change detectiON) dataset.

- **Total Image Pairs**: ~2,968 bi-temporal image pairs ($512 \times 512$ optical aerial rasters).
- **Total Question-Answer Samples**: Approximately 122,000 QA pairs across Train, Validation, and Test splits.
- **Lineage**: Constructed automatically from the SECOND dataset's semantic change segmentation masks (Wuhan University / Captain-WHU).

---

## 2. Actual Supplied File Structure
In this environment, the CDVQA repository files are located at `c:\projects\cdvqa_repo`:

```
c:\projects\cdvqa_repo/
├── Train_images.json       # 3.35 MB (Image IDs, resolution, filenames)
├── Train_questions.json    # 13.20 MB (Question text, types, image references)
├── Train_answers.json      # 6.93 MB (Ground truth answers)
├── Val_images.json         # 826 KB (6,400 image records, 400 unique image pairs)
├── Val_questions.json      # 3.25 MB (Question samples for validation)
├── Val_answers.json        # 1.71 MB (Validation answer labels)
├── Test_images.json        # 2.02 MB
├── Test_questions.json     # 7.92 MB
├── Test_answers.json       # 4.16 MB
├── Test2_images.json       # 1.96 MB
├── Test2_questions.json    # 6.35 MB
└── Test2_answers.json      # 3.27 MB
```

---

## 3. Metadata & Resolution Specifications
Each image record in `Val_images.json` contains:
```json
{
  "id": 0,
  "res_x": ".1524m",
  "res_y": ".1524m",
  "questions_ids": [0, 1, 2, 3, 4, 5],
  "file_name": "02180.png",
  "active": true
}
```

- **Ground Sample Distance (GSD)**: The spatial resolution is `.1524m` ($0.1524\text{ m} \approx 6\text{ inches}$ per pixel) representing high-resolution aerial photography.
- **Physical Ground Footprint**: For a $512 \times 512$ tile:
  $$\text{Width} = 512 \times 0.1524\text{ m} \approx 78.03\text{ m}$$
  $$\text{Ground Area} \approx 78.03 \times 78.03 \approx 6,088.5\text{ m}^2 \approx 0.6088\text{ hectares}$$
- **Verification Rule**: The application parses `res_x` dynamically from image metadata and never hardcodes 0.1524m for arbitrary inputs. If GSD metadata is absent, it reports image-relative statistics (pixel counts and percentages).

---

## 4. Question Taxonomy & Query Mapping
CDVQA classifies questions into 8 distinct types, which our platform maps cleanly to the 4 supported analytical intents:

| CDVQA Question Type | Example Question | Mapped Platform Intent |
| :--- | :--- | :--- |
| `decrease_or_not` | *"Did the areas of non-vegetated ground surface decrease?"* | `VEGETATION_LOSS` |
| `increase_or_not` | *"Have the areas of trees increased?"* | `VEGETATION_GAIN` |
| `change_ratio` | *"What is the percentage of changed areas?"* | `CHANGE_STATISTICS` |
| `change_or_not` | *"Did the areas of low vegetation change?"* | `GENERAL_COMPARISON` |
| `change_to_what` | *"What did the low vegetation change into?"* | `GENERAL_COMPARISON` |
| `smallest_change` | *"What type of change is the smallest?"* | `GENERAL_COMPARISON` |
| `largest_change` | *"What type of change is the largest?"* | `GENERAL_COMPARISON` |

---

## 5. Distinction: CDVQA Optical vs. Sentinel-2 Multispectral
- **CDVQA / SECOND Images**: 3-channel RGB imagery. They **do not** have a Near-Infrared ($B8$) band. Therefore, real NDVI **cannot** be computed on native CDVQA imagery.
- **Sentinel-2 Multi-Spectral Rasters**: 4+ band GeoTIFFs containing Red ($B4$) and Near-Infrared ($B8$) surface reflectance with projected CRS (e.g. UTM Zone 32N, EPSG:32632). These enable true physical NDVI computation and hectare/$\text{km}^2$ area reporting.
- The platform provides distinct processing branches for both modes.
