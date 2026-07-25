# Data Guide

## Paired data contract

Training expects aligned infrared and visible images with identical filename stems:

```text
dataset/
├── infrared/
│   └── scene_001.tif
└── visible/
    └── scene_001.png
```

Images may use different extensions. Visible targets are resized to the infrared spatial dimensions before an aligned random crop is taken.

## Dataset selection

Candidate public datasets must be reviewed against the intended sensor and scene domain. Ground-level multispectral driving data can validate the training system but does not substitute for satellite or airborne evaluation.

Record:

- sensor and wavelength band;
- radiometric versus display-processed status;
- spatial resolution and registration error;
- capture time and synchronization;
- location, weather, season, and time of day;
- license and redistribution restrictions;
- preprocessing already applied by the provider.

## Split strategy

Random image splitting can leak nearly identical neighboring frames. Prefer grouped splits by sequence, capture session, geography, and sensor. Maintain at least:

- training set;
- in-domain validation set;
- sensor-disjoint test set;
- geography-disjoint test set;
- adverse-condition challenge set.

## Preprocessing

Do not overwrite source files. Store transforms and statistics in a versioned manifest. Inspect alignment before training; small geometric offsets can encourage blur and penalize otherwise correct edges.

For radiometric infrared data, keep the original bit depth and calibration coefficients. The baseline display pipeline normalizes percentiles, but scientific workflows should retain temperature or radiance units separately.

## Governance

Verify that collection, processing, storage, and release are permitted by the dataset license and applicable policy. Strip unnecessary metadata before distribution and avoid collecting personally identifying imagery when it is not required for the research objective.
