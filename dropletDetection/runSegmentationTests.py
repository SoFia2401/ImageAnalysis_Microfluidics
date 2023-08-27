#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:44:29 2023

@author: daniel
"""

os.chdir("/home/sofiahernandezgelado/Documents/SuperResolution_Microfluidics")


!python dropletDetection/houghCircle_multiprocessing.py --image-folder 'data/test' --scale 2\
    --model 'srcnn'\
    --depthUm '300'\
    --depthPx '299'
!python dropletDetection/SAM_circleDetection_csv.py --image-folder 'data/test' --scale 2\
    --model 'srcnn'\
    --depthUm '300'\
    --depthPx '299'
