# GAN-based-Thoracic-Aorta-Segmentation

## Data:<br />
50 non-contrast-enhanced(NCE) CT images and 20 contrast-enhanced CTA images<br />

## Training:<br />
### Baseline:<br />
U-net was trained on 40 NCE CT images and its segmentation labels.<br /> 
### Tow-stage training:<br />
At 1st stage, CycleGAN was trained on 40 NCE CT images and 20 CTA images. At 2nd stage, the trained CylceGAN translated the CTA images to NCE CT images, than a U-net was trained on 40 NCE CT images. (set *use_end2end_training = False* in CycleGAN.py) <br />
### Hybrid network (one-stage training):<br />
Added the U-net in the CycleGAN framework and implemented end-to-end training on 40 NCE CT images and 20 CTA images. (set *use_end2end_training = True* in CycleGAN.py )<br />
  
### Testing:<br /> 
5-fold cross validation, test on 10 NCE CT images.<br />

## Results:<br />
![1](https://user-images.githubusercontent.com/55094824/131869822-542ad092-3f1c-41f4-bee6-3733e065168d.png)
![2](https://user-images.githubusercontent.com/55094824/131870415-3930a80f-f524-408d-a74f-225a57ebf2bf.png)
![3](https://user-images.githubusercontent.com/55094824/131870792-458e8a4d-60aa-4009-977b-d163e89ab147.png)



