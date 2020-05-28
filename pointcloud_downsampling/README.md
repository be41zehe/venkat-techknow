# Point Cloud Downsampling

### Dependencies
Installed inside with file with subprocess


### Requirements

Linux (Ubuntu 18.04)

virtualenv -p python3.7 $environment_name$

source $environment_name$/bin/activate

pip install -r requirements.txt

Windows

py -m pip install --user virtualenv

py -m venv $environment_name$

.\$environment_name$\Scripts\activate

pip install -r requirements.txt


### How to Run

From a terminal or cmd, Run the above lines of code from requirements


``` sh
python down_sampler.py path_to_file voxel_size
```

For Example:

``` sh
python down_sampler.py studenttask_cloud_downsampling.xyz 3
```
 
