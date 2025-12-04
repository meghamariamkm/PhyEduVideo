# PhyEduVideo: A Benchmark for Evaluating Text-to-Video Models for Physics Education

**Accepted at WACV 2026**

<p align="left">
  <a href="#-quick-start"><b>Quick Start</b></a> |
  <a href="https://meghamariamkm.github.io/phyeduvideo26/"><b>ProjectPage</b></a> |
  <a href="#"><b>arXiv</b></a> |
  <a href="#citation"><b>Citation</b></a> <br>
</p>



## 🎩Abstract

Generative AI models, particularly Text-to-Video (T2V) systems, offer a promising avenue for transforming science education by automating the creation of engaging and intuitive visual explanations. In this work, we take a first step toward evaluating their potential in physics education by introducing a dedicated benchmark for explanatory video generation. The benchmark is designed to assess how well T2V models can convey core physics concepts through visual illustrations. Each physics concept in our benchmark is decomposed into granular teaching points, with each point accompanied by a carefully crafted prompt intended for visual explanation of the teaching point. T2V models are evaluated on their ability to generate accurate videos in response to these prompts. Our aim is to systematically explore the feasibility of using T2V models to generate high-quality, curriculum-aligned educational content—paving the way toward scalable, accessible, and personalized learning experiences powered by AI. Our evaluation reveals that current models produce visually coherent videos with smooth motion and minimal flickering, yet their conceptual accuracy is less reliable. Performance in areas such as mechanics, fluids and optics is encouraging, but models struggle with electromagnetism and thermodynamics, where abstract interactions are harder to depict. These findings underscore the gap between visual quality and conceptual correctness in educational video generation. We hope this benchmark helps the community close that gap and move toward T2V systems that can deliver accurate, curriculum-aligned physics content at scale.

<img src="static/IntroFig.png" alt="overview" style="zoom:80%;" />


## 🏆 Leaderboard



| Model                | Size |  Semantic Adherence (↑) | Physics Commonsense(↑) | Motion Smoothness(↑) | Temporal Flickering(↑) | 
|----------------------|------|-------------------------|------------------------|----------------------|------------------------|
| VideoCrafter2        | 1.2  | 0.623                   | 0.497                  | 0.902                | 0.877                  | 
| CogVideoX-5b         |  5   | 0.754                   | 0.585                  | 0.983                | 0.977                  | 
| Wa2.1                |  14  | 0.832                   | 0.602                  | 0.987                | 0.982                  | 
| Video-MSG            |   5  | 0.684                   | 0.515                  | 0.995                | 0.991                  | 
| PhyT2V               |  5   | 0.778                   | 0.605                  | 0.982                | 0.974                  | 


## 🚀 Quick Start

### File Structure
- Prompts include the video generation prompt: **Prompts.json**, it also has **SA.json**, **PC-1.json**, **PC-2.json**, **PC-3.json**
- static includes figures from the paper.
- samples includes sample videos.
- scripts consist of all evaluation codes.


### Environment

```
git clone https://github.com/meghamariamkm/PhyEduVideo.git
cd PhyEduVideo
```


## 🎬Qualitative Analysis

<img src="static/Qual-1.png" alt="overview" style="zoom:80%;" />


# Citation
If you find our benchmark/code useful, feel free to leave a star and please cite our paper as follows:
```
@inproceedings{phyeduvideo_2026,
    title={PhyEduVideo: A Benchmark for Evaluating Text-to-Video Models for Physics Education},
    author={Megha Mariam K M and Aditya Arun and Zakaria Laskar and C. V. Jawahar},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year={2026},
    publisher={IEEE/CVF}
  }
```






