# MSG-STYLEGAN-TF

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


### 4.) How to run the code (Training)

### 5.) Opensourced models information (table)

### 6.) How to use pretrained models 

### 7.) How to run evaluation scripts

### 8.) Some more details about stability

### 9.) Result tables

### 10.) Qualitative examples

### 11.) Other contributors

### 12.) Thanks and regards