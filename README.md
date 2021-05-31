# Superresolution using an efficient sub-pixel convolutional neural network

이 코드는 Pytorch 공식 example안에 있는 ESPCN 코드를 활용하였습니다.
["pytorch/examples/super_resolution"]("https://github.com/pytorch/examples/tree/master/super_resolution")

## EPSCN을 활용한 Super-Resolution

논문:
["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158)  
해당 논문은 네트워크 안에서 초해상화와 같은 역할을 수행하기 위해 공간 해상도를 높이는 방법으로 ESPCN이라는 새로운 방법을 제안하였습니다.

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```

This example trains a super-resolution network on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model*epoch*<epoch_number>.pth

## Example Usage:

### Train

`python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001`

### Super Resolve

`python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`

```
calc_psnr_byinterpolation.py
line 84: calc_psnr() 함수의 입력으로 들어간 폴더 내의 영상들을 읽어들인 후,
upscale_factor = 3 만큼 영상 크기를 축소했다가 다시 확대한 후,
원 영상과의 psnr계산.
모든 영상의 평균 psnr 출력

main_syn+.py
실영상(dataset/ImageNet 폴더) 또는 합성영상(dataset/Level_Design) 폴더 영상을 이용하여 학습한 후,
실영상(dataset/BSDS300)과 합성영상(dataset/CG, dataset/valid)을 이용하여 테스트.
각 테스트 영상에 대해 평균 psnr 출력 명령어 인수 부분에 --trainCG를 추가하면
합성영상으로 학습, 삭제하면 실영상으로 학습함
(나머지 명령어 인수 부분은 line 14 ~ line 23 부분의 코드 참조)

```
