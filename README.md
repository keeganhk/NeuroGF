# NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries

This is the official implementation of **[[NeuroGF](https://arxiv.org/pdf/2306.00658.pdf)] (NeurIPS 2023)**, a learning-based framework for deep implicit representation and prediction of point-to-point geodesic distances and shortest paths of 3D shape models.

This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 on Ubuntu 20.04.

<p align="center"> <img src="https://github.com/keeganhk/NeuroGF/blob/master/imgs/example.png" width="70%"> </p>

<p align="center"> <img src="https://github.com/keeganhk/NeuroGF/blob/master/imgs/neurogf_ovft.png" width="62.5%"> </p>

### Instruction

- Download our prepared datasets of [Models](https://drive.google.com/file/d/1MLYH0C6YqVAH7XLDT0GmFAi6WF6ZWxRK/view?usp=sharing) and [ShapeNet13](https://drive.google.com/file/d/1W2I18gR4KDz19_DGYL7bxpADaZRk00wv/view?usp=sharing). The two Zip files should be decompressed and put in the ```data``` folder.

  -- (1) The dataset of ```data/Models``` contains ground-truth training and testing data of 10 popular and commonly-used 3D shape models, including: *armadillo*, *bimba*, *bucket*, *bunny*, *cow*, *dragon*, *fandisk*, *heptoroid*, *maxplanck*, *nail*.
  
  -- (2) The dataset of ```data/ShapeNet13``` collects over 25K pre-processed 3D mesh models of 13 different shape categories, including: *airplane*, *bench*, *cabinet*, *car*, *chair*, *display*, *lamp*, *loudspeaker*, *rifle*, *sofa*, *table*, *telephone*, *watercraft*. This dataset is used for the training and testing of our extension of generalizable NeuroGF learning frameworks.

- The pre-trained network parameters for both per-model overfitting and generalizable working modes are provided in ```ckpt```.

- The scripts of NeuroGF learning with the per-model overfitting working mode are provided in ```code/neurogf_ovft```. For training, one should run *pretraining_sdf_querier.py*, *training_neurogf.py*, and *separate_refining.py*, sequentially. Note that we can flexibly adjust the number of query points and fitting epochs/iterations to adapt to different requirements of the trade-off with the representation accuracy and the time/memory cost.

- The scripts of generalizable NeuroGF learning are provided in ```code/neurogf_gen```. Note that in this repository we slightly modified the network structure. The original version described in our paper directly concatenates the extracted global shape codeword with query point coordinates, while here we fetch the feature vector of the nearest neighbor of the targeted query point. We found that such simple modification produces even better performances for input point clouds with 2K points (much sparser than 8K as used in our paper).




### Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{zhang2023neurogf,
	  title={NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries},
	  author={Zhang, Qijian and Hou, Junhui and Adikusuma, Yohanes Yudhi and Wang, Wenping and He, Ying},
	  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	  year={2023}
	}

