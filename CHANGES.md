# Changes in this version (merged with your image_forensics_project2.zip)

Kept from your copy
- utils/screenshot_guard.py   (your rectilinear-edge version: fewer false "screenshot" warnings on photos)
- the lettered A/B "undetermined" overlay (draw_overlay_undetermined) and the "cannot tell original vs copy" idea

New / replaced
- utils/forensics.py          dense block-matching, new source-vs-copy border cue, mirrored-clone matching,
                              weak-match filter (<20% of strongest), adaptive texture threshold
- utils/pipeline.py           'Try another source' (pick), ambiguity (gap < 0.75), collage/figure rule,
                              network fusion (splicing / removal / retouch)
- models/inpainting_module.py exemplar (patch-copy) inpainting; huge holes use a fast fallback
- app.py                      swap button, ambiguity text
- utils/visualize.py          + lettered overlay
- data/dataset_generator.py, data/train.py, models/multi_stream_net.py, train.py, force_train.py
                              5-class training (untested: needs PyTorch)
- run.py                      dependency check

Put your own test images in outputs/ again (this zip ships it empty).


## Round 3 (after your image_forensics_project__2__1.zip)
- utils/pipeline.py: screenshot-of-a-page/figure rule (ui_score >= 0.70 -> INCONCLUSIVE), tiny-duplicate rule (< 0.35% of image -> INCONCLUSIVE)
- models/inpainting_module.py: holes larger than 4% of the image are restored at a coarser working size (32 s -> ~5 s)
- Your uploaded copy mixed old and new files: force_train.py and data/train.py were old (crash), utils/pipeline.py was the pre-network version.
  Use every file from this zip together.


## Round 4
- utils/screenshot_guard.py : find_picture_region() - finds the picture inside a screenshot
- utils/pipeline.py         : analyses the picture inside screenshots (no 'screenshot' verdict when a picture is found),
                              tiny repeats are ignored instead of INCONCLUSIVE, optional feature model wired in
- app.py                    : shows the analysed picture region
- utils/type_features.py, utils/type_model.py, train_features.py : tamper-type model WITHOUT PyTorch (scikit-learn)
- weights/experimental/feature_model.joblib : experimental model; copy to weights/feature_model.joblib to enable.
  Measured: unreliable on real photos (see README notes) - train it with real images before relying on it.
