import subprocess
import sys
import argparse

'''
Note:
In the below representation pcd is abbreviated as point_cloud
The try..finally block performs the libraries check and installs necessary dependencies
'''
try:
    import open3d as o3d
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.call([sys.executable, "-m", "pip3", "install", 'open3d'])
    subprocess.call([sys.executable, "-m", "pip3", "install", 'numpy'])
    subprocess.call([sys.executable, "-m", "pip3", "install", 'matplotlib'])
finally:
    import open3d as o3d
    import matplotlib.pyplot as plt
    import numpy as np

## Argument Parser for command line args
ap = argparse.ArgumentParser()
ap.add_argument("filename")
ap.add_argument("voxelsize")
args = ap.parse_args()

## Function read_from_file reads an input point cloud as a file and returns the object
def read_from_file(filename):
    pcd = o3d.io.read_point_cloud(filename)
    # print(pcd) # Uncomment to see the points number
    return pcd

## Function downsample_pcd performs downsampling w.r.t voxel size and returns output
def downsample_pcd(pcd, voxel_size):
    downpcd = pcd.voxel_down_sample(voxel_size)  # Downsample the point cloud with voxel size
    #print(downpcd) # Uncomment to see the points number
    return downpcd

## Function visualize_pcd displays the output pcd
def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

## Function write_to_file performs a write operation to output ASCII File
def write_to_file(filename):
    o3d.io.write_point_cloud("downsampled_point_cloud.xyz", filename, write_ascii=True)
    print("Written into downsampled_point_cloud.xyz")

def main():
    filename = args.filename
    voxel_size = float(args.voxelsize)
    input_pcd = read_from_file(filename)
    downsampled_pcd = downsample_pcd(input_pcd, voxel_size)
    visualize_pcd(downsampled_pcd)
    write_to_file(downsampled_pcd)
    

if __name__ == "__main__":
    main()


