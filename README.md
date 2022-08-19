# CAM-Decoupled-SOLO
Non-contact weight estimation system for fish based on instance segmentation

The deployment of the CAM-Decoupled-SOLOl relies on the open source code repository (Wang et al., SOLO: Segmenting Objects by Locations).

Non-contact methods for estimating cultured fish weight are essential for aquaculture companies to develop aquaculture strategies and management plans.
However, it is challenging for current technologies to precisely measure the size of fish in a densely breeding environment and estimate the weight of 
fish because of issues including occlusion, fish bending and not facing the camera. This study, which focuses on the aforementioned issues, suggests a 
technique for creating an instance segmentation dataset appropriate for measuring fish size as well as an attention-based fully convolutional instance 
segmentation network (CAM-Decoupled-SOLO) to extract fish contour features. To save labor cost and avoid fish damage, an automatic fish perimeter measurement 
model and a weight prediction system were constructed by combining fish contours extracted by CAMDecoupled-SOLO and binocular vision.  There was a 
significant correlation between fish perimeter and weight, which can be used to estimate the weight of fish in complex aquaculture environment.



## Installation
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.0.0). Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Usage

### A quick demo

Once the installation is done, you can use [inference_demo.py](demo/inference_demo.py) to run a quick demo.

### Train with multiple GPUs
    ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

    Example: 
    ./tools/dist_train.sh configs/CAM-Decoupled-SOLO/CAM-Decoupled-SOLO.py  8

### Train with single GPU
    python tools/train.py ${CONFIG_FILE}
    
    Example:
    python tools/train.py configs/CAM-Decoupled-SOLO/CAM-Decoupled-SOLO.py

### Testing
    # multi-gpu testing
    ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    ./tools/dist_test.sh configs/CAM-Decoupled-SOLO/CAM-Decoupled-SOLO.py work_dirs/fish/CAM-Decoupled-SOLO.pth  8  --show --out results_solo.pkl --eval segm

    # single-gpu testing
    python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    python tools/test_ins.py configs/CAM-Decoupled-SOLO/CAM-Decoupled-SOLO.py work_dirs/fish/CAM-Decoupled-SOLO.pth --show --out  results_solo.pkl --eval segm


### Visualization

    python tools/test_ins_vis.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --save_dir  ${SAVE_DIR}
    
    Example: 
    python tools/test_ins_vis.py configs/CAM-Decoupled-SOLO/CAM-Decoupled-SOLO.py work_dirs/fish/CAM-Decoupled-SOLO.pth --show --save_dir  work_dirs/vis_solo

## Contributing to the project
Any pull requests or issues are welcome.

