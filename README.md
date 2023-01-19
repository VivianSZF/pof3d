# Learning 3D-aware Image Synthesis with Unknown Pose Distribution

<div align=center>
<img src="./docs/assets/framework.jpg" width=600px>
</div>

Figure: Framework of PoF3D, which consists of a pose-free generator and a pose-aware discriminator. The pose-free generator maps a latent code to a neural radiance field as well as a camera pose, followed by a volume renderer (VR) to output the final image. The pose-aware discriminator first predicts a camera pose from the given image and then use it as the pseudo label for conditional real/fake discrimination, indicated by the orange arrow.

> **Learning 3D-aware Image Synthesis with Unknown Pose Distribution** <br>
> Zifan Shi*, Yujun Shen*, Yinghao Xu, Sida Peng, Yiyi Liao, Sheng Guo, Qifeng Chen, Dit-Yan Yeung <br>
> *arXiv: 2301.07702* <br>
> (* indicates equal contribution)

![image](./docs/assets/teaser.jpg)

**Figure:** Images and geometry synthesized by <b>PoF3D</b> under random views, <br><i>without any pose prior</i>.

[[Paper](https://arxiv.org/abs/2301.07702)]
[[Project Page](https://vivianszf.github.io/pof3d/)]

This work proposes *PoF3D* that frees generative radiance fields from the requirements of 3D pose priors. We first equip the generator
with an efficient pose learner, which is able to infer a pose from a latent code, to approximate the underlying true pose
distribution automatically. We then assign the discriminator a task to learn pose distribution under the supervision
of the generator and to differentiate real and synthesized images with the predicted pose as the condition. The pose-free generator and the pose-aware discriminator are jointly trained in an adversarial manner. Extensive results on a couple of datasets confirm that the performance 
of our approach, regarding both image quality and geometry quality, is on par with state of the art. To our best knowledge, PoF3D 
demonstrates the feasibility of learning high-quality 3D-aware image synthesis <i>without using 3D pose priors</i> for the first time.


## Results


Syntheses on FFHQ.
<img src="./docs/assets/ffhq1.jpg"/>


Syntheses on Cats.
<img src="./docs/assets/cats1.jpg"/>


Syntheses on ShapeNet Cars.
<img src="./docs/assets/car1.jpg"/>


## Code Coming Soon

## BibTeX

```bibtex
@article{shi2023pof3d,
  title   = {Learning 3D-aware Image Synthesis with Unknown Pose Distribution},
  author  = {Shi, Zifan and Shen, Yujun, and Xu, Yinghao and Peng, Sida and Liao, Yiyi and Guo, Sheng and Chen, Qifeng and Dit-Yan Yeung},
  journal = {arXiv:2301.07702},
  year    = {2023}
}
```
