# Crowd density estimation

This project utilises a [VGG model with Bayesian loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting) to estimate the desity of a crowd.

### Model Settings
```yaml
device: 'cpu' # Device string used for pytorch (options: 'cpu'| 'gpu')
```

### Model config

```yaml
model_path: 'weight.pth'
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
cfg:
  E: [ 64, 64, 'M', 128, 128,
      'M', 256, 256, 256, 256,
      'M', 512, 512, 512, 512,
      'M', 512, 512, 512, 512 ]
```

### Input/ Output

- Input: It receives the RGB image as a FrameObject:

  ```python
  def inference(self, frame_object: FrameObject) -> Inference
  ```
- Output: Updates the `Inference.extra` object with the  `crowd_number`

  ```python
  return Inference(extra={"crowd_number": crowd_no})
  ```
