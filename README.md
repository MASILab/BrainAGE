# BrainAGE
## Background
Context-aware prediction of chronological age from T1-weighted brain MRI. 

## Cite
The paper can be found [HERE](https://www.sciencedirect.com/science/article/pii/S0730725X1830609X?casa_token=_TmuzKk9iXAAAAAA:mzuPI1h_thyDFJ4XU2dSY9jda4s9ZVpH-DE47A5KaNv3ZPuwZ_sifJPopgmhvKu7wGZjeDsw) with the corresponding citation:

Bermudez, C., Plassard, A. J., Chaganti, S., Huo, Y., Aboud, K. S., Cutting, L. E., ... & Landman, B. A. (2019). Anatomical context improves deep learning on the brain age estimation task. Magnetic resonance imaging, 62, 70-77.

## Directions

This networks predicts chronological age using two inputs: 

1) A T1-weighted brain MRI registered to MNI-305space and intensity-normalized (see manuscript for details)
2) Volumetric estimates of 132 brain regions following the BrainCOLOR protocol. I reccommend using [SLANT](https://github.com/MASILab/SLANTbrainSeg) to obtain this automated segmentation. 
    
  



