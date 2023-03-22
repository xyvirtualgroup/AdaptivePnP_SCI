


# Adaptive Deep PnP Algorithm for Video Snapshot Compressive Imaging (AdaptivePnP_SCI)
This repository contains the python code for the paper **Wu, Z., Yang, C., Su, X., & Yuan, X. (2022). Adaptive Deep PnP Algorithm for Video Snapshot Compressive Imaging. arXiv preprint arXiv:2201.05483.**.

The journal paper **Wu, Z., Yang, C., Su, X., & Yuan, X. (2023). Adaptive Deep PnP Algorithm for Video Snapshot Compressive Imaging. International Journal of Computer Vision, https://doi.org/10.1007/s11263-023-01777-y.** has been published in March, 2023. The latest version code will be updated soon.


## Usage
### Requirements
```
numpy==1.21.2
pytorch==1.8.1
scipy==1.7.1
scikit-image==0.18.1
h5py==2.10.0
tqdm==4.61.2
tensorboardx==2.2
```

### Data
Please add simulation middle scale color data from [PnP_SCI](https://github.com/liuyang12/PnP-SCI_python) to ./dataset.

Our results are saved in [OneDrive Link](https://westlakeu-my.sharepoint.com/:f:/g/personal/wuzongliang_westlake_edu_cn/EsLJ8rbIci1AoZYUgBUcNMoBV4IigjxHVc6NbddjACnitg?e=NCiUMm).  



### Test

1. Run ```ADMM_TV_Warm_Start_save.py``` to save the TV prior initialized results.
[OPTION] Or just add the TV prior initialized results (```_Admm_tv_xxx_bayer8.mat```) from [OneDrive Link](https://westlakeu-my.sharepoint.com/:f:/g/personal/wuzongliang_westlake_edu_cn/EsLJ8rbIci1AoZYUgBUcNMoBV4IigjxHVc6NbddjACnitg?e=NCiUMm) to ```./results/savedmat/```.  

2. Run ```twoStageADMM_Online_FFD_WARM.py``` or ```twoStageADMM_Online_FastDVD_WARM.py``` to test the algorithm after loading results initialized with TV prior as warn start.




## Structure of directories

| directory  | description  |
| :--------: | :----------- | 
| `dataset` | data original  | 
| `packages` and `models`   | algorithms pluged into PnP framework|
| `model_zoo`   | pre-trained model data|
| `dataset`    | data used for reconstruction (simulated or real data) |
| `results`    | results of reconstruction (after reconstruction) |
| `utils`      | utility functions |



## Citation
```
@article{wu_adaptive_2023,
 title = {Adaptive {Deep} {PnP} {Algorithm} for {Video} {Snapshot} {Compressive} {Imaging}},
 issn = {1573-1405},
 url = {https://doi.org/10.1007/s11263-023-01777-y},
 doi = {10.1007/s11263-023-01777-y},
 journal = {International Journal of Computer Vision},
 author = {Wu, Zongliang and Yang, Chengshuai and Su, Xiongfei and Yuan, Xin},
 month = mar,
 year = {2023},
}
```



