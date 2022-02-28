# mmdet_practice



## Install

Follow [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) from [mmdetection](https://github.com/open-mmlab/mmdetection)



> MUST INSTALL CONDA FIRST!
>
> If you install it later..... Try..



## High-level APIs for inference



First, Here is an example of building the [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) model and Inference.



1. Prepare cfg file



Download cfg file from [mmdetection github](https://github.com/open-mmlab/mmdetection) or clone it



2. And weight file



Download [it](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and move it to "**checkpoints**" directory



3. And test img



Just prepare for Inference



![test](https://user-images.githubusercontent.com/45679609/155476759-07e6687b-0fc8-4826-b298-001f3fa9e018.jpg)


and



```bash
python test.py
```



```python
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
```



then you can have `result.jpg`



![result](https://user-images.githubusercontent.com/45679609/155476799-ea0078f7-7df6-44a4-9a27-f4aeb31b3b8d.jpg)


## cfg 파일 톺아보기



`mmdetection`이라는 프레임워크에서는 많은 모델들을 지원합니다.



위의 `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`라는 파일명만으로 어떠한 backbone,neck,head를 사용했는지 알 수 있습다.



위의 cfg파일을 보면 다음과 같습니다.



```python
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

```



`mmdetection`의 가장 기본적인 컴포넌트 타입은 `config/_base_`아래에 1.`dataset`, 2.`model`, 3.`schedule`, 4.`default_runtime`가 있습니다. 이 안에 있는 config들을 **primitive**하다고 합니다.



doc에서는 오직 **한개의 **primitive를 사용하는 것을 권장하고 있습니다. 다른 config들은 primitive로부터 상속받아야 하며 **maximum level은 3입니다.**



### Config Name Style



cfg파일의 네이밍 컨벤션이 존재하는데 다음과 같은 규칙을 따릅니다.



```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```



`{xxx}` 는 필수 적으로 지정하고 `[yyy]`는 옵션입니다.

- `{model}`: `faster_rcnn`, `mask_rcnn`과 같은 모델 타입명입니다.
- `[model setting]`: `without_semantic` for `htc`, `moment` for `reppoints`와 같은 특정모델의 값입니다.
- `{backbone}`: `r50` (ResNet-50), `x101` (ResNeXt-101)과 같은 백본 타입입니다.
- `{neck}`:  `fpn`, `pafpn`, `nasfpn`, `c4`과 같은 neck 타입입니다.
- `[norm_setting]`: `bn` (Batch Normalization) 은 default값이며 다른 norm layer `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization)이 존재합니다. `gn-head`/`gn-neck` 는 head나 neck에만 적용된 것을 의미합니다.
- `[misc]`: 여타 세팅이나 플러그인을 지정합니다., e.g. `dconv`, `gcb`, `attention`, `albu`, `mstrain`.
- `[gpu x batch_per_gpu]`: GPU와 GPU per Sample로  `8x2` 가 기본값이다.
- `{schedule}`: schedule을 지정한다. `1x`, `2x`, `20e` 등의 옵션이 있는데 `1x` 과`2x`는 각각 12/24 epoch을 의미한다. `20e`는 cascade 모델에서 채용하는데 이는 20epoch을 의미한다. 1x/2x는 각각 8/16, 11/22 epoch에서 10분의 1로 줄어든다. `10e`는 16, 19 epoch에서 10분의 1로 줄어든다.
- `{dataset}`:  `coco`, `cityscapes`, `voc_0712`, `wider_face`과 같은 데이터셋 명을 지정하는 부분이다.



따라서, 우리가 예시로 사용한 `faster_rcnn_r50_fpn_1x_coco.py`라는 파일명을 통해



`Fast R-CNN` model, `ResNet-50`을 backbone으로, `fpn` Neck, 12 epoch마다 lr이 10분의 1로 줄어드는 scheduler로 사용하는 것을 유추할 수 있습니다.



### Model



![dad36fb7-2217-4d89-afd8-da12286a7d3d](https://user-images.githubusercontent.com/45679609/155488012-72c76528-0e19-481d-b6a0-89f3cce0d2de.png)



<br/>

<br/>



![download](https://user-images.githubusercontent.com/45679609/155487994-5a97430d-4b55-4568-b909-53725c6bb63e.png)



<br/>

<br/>

![download (1)](https://user-images.githubusercontent.com/45679609/155488049-5a584829-2dc6-462d-bb8e-5693b525b591.png)

<br/>

<br/>


`faster_rcnn_r50_fpn.py`로 모델을 톺아보겠습니다. 설명은 모두 주석으로 달았습니다.



```python
# model settings
model = dict(
    type='FasterRCNN', # 해당 모델의 대표이름을 지정
    # ============================== [backbone] ============================== #
    backbone=dict( 
        type='ResNet', #ResNet을 사용
        depth=50, #ResNet-50을 사용, ResNet계통의101도 존재
        num_stages=4, #ReNet의 stem쪽인 stage개수를 의미
        out_indices=(0, 1, 2, 3), # 각 스테이지에서의 feature map output index를 의미
        frozen_stages=1, # freezing시킬 스테이지를 지정, 여기서는 1번째 stage를 freeze
        norm_cfg=dict( # Normalize의 cfg를 적용
            type='BN', # norm layer는 BN, GN(Group Norm)이 존재
            requires_grad=True), # BN에서의 gamma, beta를 학습에 포함시킬지 여부
        norm_eval=True, # evaluation할 때 freeze시킬 것인지 여부
        style='pytorch', # backbone의 style로 'pytorch'는 3x3 conv s=2, 'caffe'는 1x1 conv s=2 의미
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # ImageNet으로 pretrain된 것을 																					load
    # ============================== [neck] ============================== #
    neck=dict(
        type='FPN', # FPN, NASFPN, PAFPN을 지원
        in_channels=[256, 512, 1024, 2048], # input channel로 backbone의 output channel과 동일해야함
        out_channels=256, #각 FPN feature map의 output channel
        num_outs=5), # output scale
    # ============================== [head] ============================== #
    # ------------------------------ [RPN head] ------------------------------ #

    rpn_head=dict(
        type='RPNHead', # RPN, GARPNHead 지원
        in_channels=256, # input channel로 backbone의 output channel과 동일해야함
        feat_channels=256,# head에서 Conv layer의 feature channel
        anchor_generator=dict(
            type='AnchorGenerator', # 대부분의 method는 AnchorGenerator를 사용
            scales=[8], # anchor의 scale. anchor의 크기는 scale * base_sizes가 됨
            ratios=[0.5, 1.0, 2.0], #anchor의 width, height 비율로 총 3가지의 비율이 존재
            strides=[4, 8, 16, 32, 64]), # anchor generator의 stride로 FPN과 동일
        bbox_coder=dict( # bbox를 train,test할 때, encode, decode하는 cfg
            type='DeltaXYWHBBoxCoder', # 대부분의 method에서 채용함
            target_means=[.0, .0, .0, .0], # mean
            target_stds=[1.0, 1.0, 1.0, 1.0]), # variance
        loss_cls=dict( # cls branch에서 사용할 loss function cfg
            type='CrossEntropyLoss', # Focal Loss도 지원
            use_sigmoid=True, # RPN은 객체의 유무만 판단하는 binary clssification이므로 sigmoid사용
            loss_weight=1.0), # cls branch의 loss가중치
        loss_bbox=dict(type='L1Loss', # reg branch의 cfg로 IoU, Smooth L1등 지원
                       loss_weight=1.0)), # reg branch의 loss가중치
    # ------------------------------ [ROI head] ------------------------------ #
roi_head=dict(
        type='StandardRoIHead', # ROI 헤드의 종류
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor', # ROI feature extractor로 대부분 SingleRoIExtractor사용
            roi_layer=dict(type='RoIAlign', # ROI layer의 종류로 다양한 종류 지원
                           output_size=7, # feature map의 output사이즈
                           sampling_ratio=0), # extract할때 sample ratio로 0은 adaptive를 의미
            out_channels=256, # output channel
            featmap_strides=[4, 8, 16, 32]), # feature map의 stride로 backbone과 같아야함
        bbox_head=dict(
            type='Shared2FCBBoxHead', # bbox head의 종류
            in_channels=256, # bbox head의 input으로 roi_extractor의 output channel과 동일해야함
            fc_out_channels=1024, # FC Layer의 output channel
            roi_feat_size=7, # ROI feature의 사이즈
            num_classes=80, # classification할 class 개수
            bbox_coder=dict( 
                type='DeltaXYWHBBoxCoder', #second stage에서 사용하는 coder
                target_means=[0., 0., 0., 0.], # mean
                target_stds=[0.1, 0.1, 0.2, 0.2]), # variance로 box의 location이 정확해야하므로 상대적으로 낮음
            reg_class_agnostic=False, # regression이 class agnostic한지 여부 ( 잘 모 르 겠 음 )
            loss_cls=dict( # RPN head부분의 것과 동일
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
 	# ============================== [Train] ============================== #
    train_cfg=dict(
        rpn=dict(
            assigner=dict( # assigner란 각각의 bbox를 gt 나 bg에 할당하는 역할
                type='MaxIoUAssigner', # 가장 많이쓰는 타입
                pos_iou_thr=0.7, # 0.7보다 높을 경우 positive로 할당
                neg_iou_thr=0.3, # 0.3보다 낮을 경우 negative로 할당
                min_pos_iou=0.3, # positive 중 pos_iou_thr보다 낮은 경우가 4th step때 생길 수 있어 최소값설정
                match_low_quality=True, # 한 BBOX가 여러개의 cls에 높은 score를 가질 때, 낮은 score에도 할당하는 옵션
                ignore_iof_thr=-1), # Iof threshold로 https://github.com/open-mmlab/mmdetection/issues/393#issuecomment-472282509해당 링크에서 의미 확인
            sampler=dict( # sampler란 위의 할당된 bbox들을 sample로 만드는 과정
                type='RandomSampler', # random하게 sample을 뽑는 method
                num=256, # sample의 개수
                pos_fraction=0.5, # positive의 비율
                neg_pos_ub=-1, # sample들의 상한선(upper bound)로 default = 0
                add_gt_as_proposals=False), # gt를 proposal로 추가할지여부 default = True
            allowed_border=-1, # anchor의 경계를 허용시키는 값으로 0보다 큰값일 경우 anchor에 더해짐. -1은 모두 valid.default=0
            pos_weight=-1, # positive sample에 대한 가중치
            debug=False), # debug모드 설정
        rpn_proposal=dict(
            nms_pre=2000, # NMS 전의 bbox개수
            max_per_img=1000, # NMS 이후의 최대 bbox 개수
            nms=dict(type='nms', iou_threshold=0.7), # NMS type
            min_bbox_size=0), # 최소 크기 설정
        rcnn=dict( # roi head
            assigner=dict( #위의 rpn과 동일
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    # ============================== [Test] ============================== #
    test_cfg=dict( # train과 동일
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

```



위의 모델에 대한 cfg파일로는 구성요소와 input, output을 확인할 수 있을 뿐 내부의 실질적인 구조에 대해서는 알 수 없습니다.



따라서, 해당 module의 세부사항을 알기 위해서는 `type`에 적힌 module을 모두 살펴보아야합니다.

[mmdet docs](https://mmdetection.readthedocs.io/en/latest/index.html)



