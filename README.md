<h1> 3D-DaVa Package </h1>

3D-DaVa is a Python-based package that offers point cloud (3D) data validation for an array of industrial products, allowing  point cloud quality assessment using no-reference and reference-based methods. Compared to other quality assessment tools, which often output use-case specific metrics, 3D-DaVa can offer a more general approach, covering three data quality dimensions. Accuracy quantifies the presence of noise. Validity quantifies the presence of outliers. Completeness quantifies the absence of expected values. 3D-DaVa provides a couple other functionalities, such as stitching point clouds together and denoising point clouds. A non-automatic setting of the pipeline allows user involvement, not only by applying a weighing scheme to highlight desired properties in an overall quality score, but also to refine the assessment process. 

Implementation backbone: Open3D


<h2> Installation Guide </h2>

To use 3D-DaVa, first follow the installation steps:

Anaconda and Git are required prior to package installation !

Installation steps:

1. Via Git: clone repository locally using either the HTTPS or SSH-key using the command:

    git clone <ADD HTTPS/SSH-KEY>

2. You should now have a ProjectClouds folder containing all necessary files. Change location to the ProjectClouds folder through your Anaconda Prompt terminal.

3. In the Anaconda Prompt, create the conda environment. This can be done in two ways, we recommend alternative 2 while the package is under work.

Alternative 1. Create environment from the environment.yml file by entering:

    conda env create -f environment.yml

Alternative 2. Create a new environment by entering:

    conda create -n 3d-dava python=3.9

In the new environment, manually pip install the following packages: open3d, matplotlib, scikit-learn, scipy, pandas.

4. Verify environment creation by entering the line below. 3d-dava should be listed:

    conda env list

5. Activate environment by entering:

    conda activate 3d-dava

6. Whilst in the ProjectClouds directory, run the following to install 3D-DaVa package:

    python3 -m pip install .

7. The 3D-DaVa package should now be locally installed. You can verify this by checking if 3d-dava is present in the activated environment by entering:

    conda list


<h2> Usage </h2>

You can use the package directly from the command line, as shown below.

<h3> Automatic Point Cloud Assessment </h3>

The command line below shows how to run the pipeline using both the no-reference and the reference-based methods, in an automatic manner that does not require user intervention. The digitized point cloud and the reference file are the only necessary inputs, here exemplified by point_cloud.pcd and reference.stl, however the names might vary (make sure they coincide with the files!). Returns a .json file containing the accuracy, validity and completeness scores.  

    python -m 3D_DaVa processing [-h] point_cloud.pcd [-r reference.stl] -auto

<h3> Non-automatic Point Cloud Assessment </h3>

The command line below shows how to run the pipeline using both the no-reference and the reference-based methods, in an manner that does require user intervention and keeps user in the loop. The digitized point cloud and the reference file are the only necessary inputs, here exemplified by point_cloud.pcd and reference.stl, however the names might vary (make sure they coincide with the files!). Returns a .json file containing the accuracy, validity, completeness and quality scores.  

    python -m 3D_DaVa processing [-h] point_cloud.pcd [-r reference.stl]


Arguments:

**cloud_file**: point cloud file to be asessed (recommended format: .ply, see notes).

**-h, –help**: (Optional) show arguments

**-vis, --visualize**: (Optional) allow visualization of process steps

**-snap, --snapshot**: (Optional) allow snapshotting of process steps

**-r REFERENCE**, –reference REFERENCE: reference file (recommended format: .stl, see notes)

**-o OUTPUT, –output OUTPUT**: (Optional) filename of metric-filled output file. No file extension needed (automatically turned to .json)

**-save, –-save**: (Optional) allow saving the denoised point cloud as a (.pcd file) on current path. (Obs: still not implemented)

**-auto, –-automatic**: (Optional) runs the pipeline without user intervenstion.


<h3> Stitching </h3>

Fuses together several point clouds into a single point cloud, if necessary. This is a typical step when we have more digitized point clouds of the same object. For now we assume that the scale and orientation are the same for all point clouds involved.

    python -m 3D_DaVa stitching [-h] [-o OUTPUT] directory_path

Arguments:

**directory_path**: absolute path of directory containing point clouds (recommended format: .ply, see notes).

**-h, –help**: (Optional) show arguments

**-o OUTPUT, –output OUTPUT**: (Optional) name of stitched point cloud file

<h3> Datasets and Distortion Simulation </h3>
 
For experimental or evaluation purposes, one may create simulated datasets from reference files. In our experiments we have used reference models obtained from https://www.dimelab.org/fabwave. This is a repository over 3D Product model data.

Another method that has been used in our experiments is creating our own reference models. To this purpose, we have used the create_simple_geometries.py file in the Test_CAD folder, which creates simple geometries (cube, cylinder and sphere) that can be used as reference models in the distortion simulation process. To create these files, make sure to have environment above active and simply run the following in the terminal:

    python create_simple_geometries.py

To simulate distortions onto the reference files, we add all of the relevant .stl files (i.e reference models) inside the Test_CAD folder, and run the distortion simulation file (called distort.py) by simply running the following line in the terminal, after activating the environment above:

    python distort.py 

Note that the distortion solution process is stochastic.  

Note that our experiments have also included company data, however we are not able to share these due to confidentiality reasons. 