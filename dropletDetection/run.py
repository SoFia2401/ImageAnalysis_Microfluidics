import os
os.chdir("/home/sofiahernandezgelado/Documents/SuperResolution_Microfluidics")

# For analysis
!python dropletDetection/CHT.py --image-folder 'data/test' --scale 4\
    --model 'msrn'\
    --depthUm '300'\
    --depthPx '299'



!python dropletDetection/SAM_CHT.py --image-folder 'data/test' \
    --depthUm '300'\
    --depthPx '299' \
    --gpu
