## Deep Learning for Microfluidic Imaging

## Droplet detection models

To run the droplet detection models run the file /dropletDetection/run.py or open a terminal and run the following: 


### Circular Hough Transform
```
python dropletDetection/CHT.py --image-folder 'data/test' --scale 2\
    --model 'srcnn'\
    --depthUm '300'\
    --depthPx '299'
```

### Segment Anything + Circular Hough Transform
```
python dropletDetection/SAM_CHT.py --image-folder 'data/test' --scale 2\
    --model 'srcnn'\
    --depthUm '300'\
    --depthPx '299'
```

## SRCNN

To run the SRCNN model run the file /SRCNN/run.py or open a terminal and run the following: 

### Prepare datasets:

```
python /SRCNN/prepare.py\
    --images-dir "data/train"\
    --output-path "data/train/train_file_SRCNN.h5"\
    --scale 2

python /SRCNN/prepare.py\
--images-dir "data/eval"\
--output-path "data/eval/eval_file_SRCNN.h5"\
--scale 2 --eval
```


### Training
```
!python  /SRCNN/train.py\
    --train-file "data/train/train_file_SRCNN.h5" \
                --eval-file "data/eval/eval_file_SRCNN.h5" \
                --outputs-dir "SRCNN/outputs" \
                --scale 2 \
                --lr 1e-5 \
                --batch-size 32\
                --num-epochs 100\
                --num-workers 2 \
                --seed 123
 ```

### Testing 
```                
python /SRCNN/test.py\
--image-dir "data/test"\
--weights-file "SRCNN/LargeTrial/Checkpoints and results/x2/best.pth"\
--scale 2
```


## MSRN-BAM

To run MSRN-BAM first prepare your datasets. Run the following in terminal

### Prepare datasets:
```
python MSRN/prepare.py\
    --images-dir "data/train"\
    --output-path "data/train/train_file_MSRN.h5"\
    --scale 2


python /home/sofiahernandezgelado/Documents/MSRN/prepare.py\
--images-dir "data/eval"\
--output-path "data/eval/eval_file_MSRN.h5"\
--scale 2 --eval

```

### Train MSRN-BAM

To train MSRN-BAM run the file MSRN/train.py and adjust parameters as required.

### Test MSRN-BAM

To test MSRN-BAM run the file MSRN/test.py and adjust the path of the checkpoint you want to use as required. 


