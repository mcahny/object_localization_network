
# Learning Open-World Object Proposals without Learning to Classify

## Pytorch implementation for "Learning Open-World Object Proposals without Learning to Classify" ([arXiv 2021](https://arxiv.org/abs/2108.06753)) <br/>

[Dahun Kim](https://mcahny.github.io/), [Tsung-Yi Lin](https://scholar.google.com/citations?user=_BPdgV0AAAAJ), [Anelia Angelova](https://scholar.google.co.kr/citations?user=nkmDOPgAAAAJ), [In So Kweon](https://rcv.kaist.ac.kr), and [Weicheng Kuo](https://weichengkuo.github.io/).


## Introduction

Humans can recognize novel objects in this image despite having never seen them  before. “Is it possible to learn open-world (novel) object proposals?” In this paper we propose **Object Localization Network (OLN)** that learns localization cues instead of foreground vs background classification. Only trained on COCO, OLN is able to propose many novel objects (top) missed by Mask R-CNN (bottom) on an out-of-sample frame in an ego-centric video.

<img src="./images/epic.png" width="300"> <img src="./images/oln_overview.png" width="600"> <br/>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{kim2021oln,
  title={Learning Open-World Object Proposals without Learning to Classify},
  author={Kim, Dahun and Lin, Tsung-Yi and Angelova, Anelia and Kweon, In So and Kuo, Weicheng},
  journal={arXiv preprint arXiv:2108.06753},
  year={2021}
}
```
