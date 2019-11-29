# MSG-STYLEGAN-TF
## Official code repository for the paper "MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks" [[arXiv]](https://arxiv.org/abs/1903.06048)
<p align="center">
<img alt="Teaser Diagram" src="https://github.com/akanimax/msg-stylegan-tf/blob/master/diagrams/teaser.jpg" />
<br>
</p>

### Why this repository?
Our previous research work released the 
[BMSG-GAN](https://github.com/akanimax/BMSG-GAN) code in PyTorch 
which applied our proposed multi-scale connections in the basic
ProGAN architecture (i.e. DCGAN architecture) 
instead of using the progressive growing.
This repository applies the Multi-scale Gradient connections 
in StyleGAN replacing the progressive growing used 
for training original StyleGAN. The switch to Tensorflow was 
primarily to ensure an apples-to-apples comparison with StyleGAN. 

### Due Credit
This code heavily uses NVIDIA's original 
[StyleGAN](https://github.com/NVlabs/stylegan) code. We accredit and
acknowledge their work here. The 
[Original License](https://github.com/akanimax/msg-stylegan-tf/blob/master/LICENSE_ORIGINAL.txt) 
is located in the base directory (file named `LICENSE_ORIGINAL.txt`).

### Abstract
While Generative Adversarial Networks (GANs) have seen huge 
successes in image synthesis tasks, they are notoriously 
difficult to adapt to different datasets, in part due 
to instability during training and sensitivity to hyperparameters. 
One commonly accepted reason for this instability is
that gradients passing from the discriminator to the 
generator become uninformative when there isn’t enough 
overlap in the supports of the real and fake distributions. In 
this work, we propose the Multi-Scale Gradient Generative 
Adversarial Network (MSG-GAN), a simple but effective technique 
for addressing this by allowing the flow of
gradients from the discriminator to the generator at 
multiple scales. This technique provides a stable approach for
high resolution image synthesis, and serves as an alternative 
to the commonly used progressive growing technique.
We show that MSG-GAN converges stably on a variety of
image datasets of different sizes, resolutions and domains,
as well as different types of loss functions and architectures,
all with the same set of fixed hyperparameters. 
When compared to state-of-the-art GANs, our approach matches or
exceeds the performance in most of the cases we tried.

### Method overview

<p align="center">
<img alt="Architecture diagram" src="https://github.com/akanimax/msg-stylegan-tf/blob/master/diagrams/architecture_horizontal.jpg" />
<br>
</p>

Architecture of MSG-GAN, shown here on the base model proposed in 
ProGANs. Our architecture includes connections from the intermediate 
layers of the generator to the intermediate layers of the 
discriminator. Multi-scale images sent to the discriminator 
are concatenated with the corresponding activation volumes 
obtained from the main path of convolutional layers followed by a 
combine function (shown in yellow).

#### StyleGAN Modifications:
The MSG-StyleGAN model (in this repository) uses all the
modifications proposed by StyleGAN to the ProGANs architecture 
except the mixing regularization. Similar to 
MSG-ProGAN (diagram above), we use a 1 x 1 conv layer to obtain 
the RGB images output from every block of the StyleGAN generator 
leaving everything else (mapping network, non-traditional input and 
style adaIN) untouched. The discriminator architecture is same as 
the ProGANs (and consequently MSG-ProGAN) discriminator.

### How to run the code (Training)
Training can be run in the following 3 steps:

##### Step 1: Data formatting
The MSG-StyleGAN training pipeline expects the dataset to be in 
`tfrecord` format. This sped up the training to a great extent.
Use the `dataset_tool.py` tool to generate these tfrecords from your 
raw dataset. In order to use the tool, either select from the bunch 
of datasets that it already provides or use the `create_from_images`
option if you have a new dataset in the form of images. 
For full options and more information run:

    (your_virtual_env)$ python dataset_tool.py --help

##### Step 2: Run the training script
First step is to update the paths in the global configuration
located in `config.py`. For instance:
    
    """Global configuration."""
    
    # ----------------------------------------------------------------------------
    # Paths.
        
    result_dir = "/home/karnewar/self_research/msg-stylegan/"
    data_dir = "/media/datasets_external/"
    cache_dir = "/home/karnewar/self_research/msg-stylegan/cache"
    run_dir_ignore = ["results", "datasets", "cache"]
    
    # ----------------------------------------------------------------------------

The `result_dir` is where all the trained models, training logs and 
evaluation score logs will be reported. The `data_dir` should 
contain the different datasets used for training 
under separate subdirectories, while the `cache_dir` stores any 
repeatedly required objects in the training. For instance the
Mean and Std of the real images while calculating the FID.

Following this, download the inception net weights from 
[here](https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn)
and place them in `result_dir + "/inception_network/inception_v3_features.pkl"`.

Finally, modify the configurations in the `train.py` as per your 
situation and start training by just running the `train.py` script.

    (your_virtual_env)$ python train.py

### Opensourced models

### 6.) How to use pretrained models 

### 7.) How to run evaluation scripts

### 8.) Some more details about stability

### 9.) Result tables

### 10.) Qualitative examples

### 11.) Other contributors

### 12.) Thanks and regards