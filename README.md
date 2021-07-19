### Model Settings
```yaml
device: 'cpu'
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
