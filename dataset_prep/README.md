# Dataset assembly walkthrough

This is a quick map of how to build a helmet-detection dataset that won't
embarrass you in the demo. The full step-by-step lives in
[../INSTRUCTIONS.md](../INSTRUCTIONS.md) §3 — this is the short version.

## Three classes

- `rider`   — the person on the bike (any pose)
- `helmet`  — head clearly enclosed by a real helmet
- `no_helmet` — bare head, **plus hard negatives**: caps, turbans, hoodies, hats

`no_helmet` must include hard negatives or your model will call every cap a helmet.

## Sources, in order of effort vs. reward

1. **Kaggle "Helmet Detection" datasets** — fastest start, but check labels.
2. **Roboflow Universe** — search "helmet detection"; many ready-to-train YOLO datasets.
3. **Your own clips** — phone videos of your own street, then `extract_frames.py`,
   then label in Roboflow Annotate. This is what makes your demo specific.

Mix all three. Goal: ≥ ~1500 labeled images with class balance roughly
`rider ≈ helmet ≈ no_helmet`, with hard negatives at ~15–20% of `no_helmet`.

## Frame extraction from your own videos

```bash
python dataset_prep/extract_frames.py my_clip.mp4 dataset_raw/ 10
```

`10` = save every 10th frame. Lower for short clips, higher for long ones.

## Labeling in Roboflow Annotate

1. Create a project, set classes to exactly `rider`, `helmet`, `no_helmet`.
2. Upload images, label them, generate a version.
3. Augmentations to enable on the version:
   - Flip: horizontal
   - Brightness: ±25%
   - Blur: up to 1.5px
   - Noise: up to 1% of pixels
4. Export → YOLOv8 format → Download zip.

## Final folder shape (after Roboflow export)

```
dataset/
├── data.yaml
├── train/images/   train/labels/
├── valid/images/   valid/labels/
└── test/images/    test/labels/
```

Zip this folder, upload to Google Drive, then follow INSTRUCTIONS.md §4
for Colab training.
