


# Adaptive Deep PnP Algorithm for Video Snapshot Compressive Imaging (AdaptivePnP_SCI)
This repository contains the python code for the paper **Adaptive Deep PnP Algorithm for Video Snapshot Compressive Imaging**.

## Usage
### Requirements
```
Python==3.9.4
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
To be updated.
```



