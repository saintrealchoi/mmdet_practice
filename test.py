from mmdet.apis import init_detector, inference_detector
import mmcv

# 1. CFG) Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 2. Build) build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 3. Inference) test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# 4. Visualize) visualize the results in a new window
model.show_result(img, result)

# 4-1. Save)save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')