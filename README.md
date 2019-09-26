# lidar-tree-detection
LiDar Tree Detection 

LiDar Tree detection algorithm for both TLS and ALS sensor system.

Install required packages running pip install -r requirements.txt in the folder.
If you are processing LiDar files (.laz), you may need to install lastools through conda packages system.

conda install -c conda-forge lastools

In order to process an ALS LiDar file, go to main folder and run:

python detection-als.py [LiDar File]

If you want to process a TLS LiDar file run:

python detection-tls.py [LiDar File]

The script will return a CSV file with lat/lon position of every tree in the file.
