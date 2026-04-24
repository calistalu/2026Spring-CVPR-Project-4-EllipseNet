# 2026Spring-CVPR-Project-4-EllipseNet

EllipseNet extends YOLOv1 to detect oriented ellipses.

## Evaluation Protocol

During evaluation, each grid cell predicts ellipse parameters  
$(x, y, a, b, \theta)$, an objectness score, and class probability.  

We use:

$$
s = p_{\text{obj}} \cdot p_{\text{cls}}
$$

as the final confidence.

We first discard low-confidence predictions with a threshold  
(`conf_thres = 0.25` in our main runs).  
The remaining predictions are converted back to image coordinates and clipped to valid image bounds.

Then we apply Non-Maximum Suppression (NMS) class-wise:  
predictions are sorted by confidence, and lower-scored predictions are removed if their overlap with a higher-scored prediction exceeds  
`nms_iou_thres = 0.5`.  

For overlap, we use IoU computed on the predicted ellipse regions.

After NMS, the remaining detections are matched to ground-truth ellipses for metric computation.  
We report precision, recall, and mAP at IoU = 0.5.
