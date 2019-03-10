# Inpaint Faces using DCGAN
This work in this repository is based on the paper [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/pdf/1607.07539.pdf) by Yeh _et al_.

We use the same pretrained DCGAN (trained on [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)), but we used the [LFW dataset](http://vis-www.cs.umass.edu/lfw/) and our own images.

## Overview
There are 2 main components to this repo, a Jupyter Notebook tutorial / experiment and a Graphical User Interface for face inpainting. We tried some different parameters from the paper, and visualized the final result as well as the images generated after each iteration. Some results that we display are from fewer iterations than the original paper.


## How to use
First, please install the dependencies using `pip install -r requirements.txt`.

#### Optional pre-processing step if you want to use your own images
We tested on our own images after preprocessing the faces using OpenFace. The face images must be 64 x 64 RGB images.
To use OpenFace for preprocessing, you need to have Python 2. You can create an environment with Python 2, and do the following terminal commands:

1. `git clone https://github.com/cmusatyalab/openface.git`

```python
cd openface
pip2 install -r requirements.txt
python2 setup.py install
models/get-models.sh
cd ..
```


2. `./openface/util/align-dlib.py data/FolderImagesToProcess align innerEyesAndBottomLip data/NameFolderToSaveProcessedImages --size 64`

The first folder path should be changed to the folder where your images are saved. The second folder path is where processed images will be saved.


#### Inpainting GUI
To run the GUI, cd into the Inpaint_GUI folder and run `./gui_inpaint.py` or `python gui_inpaint.py`.

Option to save images after each iteration: in the gui_inpaint.py file, set self.saveEaItr = True.



*Screenshot of GUI*:

<img src="https://github.com/nlune/DCGAN-Face-Inpainting/blob/master/images/display/gui_screenshot.png" width="500"/>



*Sample of images generated after some iterations and the inpainted face after 50 iterations*:

<img src="https://github.com/nlune/DCGAN-Face-Inpainting/blob/master/images/display/traverse_manifold.png" width="500"/>



External code was used for loading the .pb file, postprocessing / Poisson blending (link to original in code). The instructions for OpenFace were taken from https://bamos.github.io/2016/08/09/deep-completion/
