
import os 
os.chdir('/home/Documents/SuperResolution_Microfluidics/DnCNN')
!python main_train_dncnn.py


!python main_test_dncnn.py\
--model_name 'denoising_sigma_3'\
--testset_name 'test_sigma6'\
--noise_level_img 3\
--show_img False