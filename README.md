# TODO
---
- [x] `requirements.txt`
- [x] multiprocessing of `activ_dump_translator.py`
- [x] multiprocessing of `top_actiated_sentences.py`
- [x] Visualize concept alignment results
- [] pretrained models/results download
- [] add kind comments to all python files.
- [x] All-in-one script
- [x] Coloring concept occurence
- [] Release Test
- [] docker install...?
- [x] Select visualization samples

# Overview
---

<img src="figures/teaser.png" width="800px" align="right">


This repository contains an implementation of our ICLR 2019 paper;

* [Seil Na](seilna.github.io), [Yo Joong Choe](https://yjchoe.github.io/), [Dong-Hyun Lee](https://scholar.google.com/citations?user=Iw-G2qIAAAAJ&hl=en) and [Gunhee Kim](http://vision.snu.ac.kr/~gunhee/). [Discovery of Natural Language Concepts in Individual Units of CNNs](https://openreview.net/forum?id=S1EERs09YQ)

**TL;DR**: Individual units of deep CNNs learned in NLP tasks (e.g. translation, classification) could act as a natural language concept detector.

This work covers the interpretability of Deep Neural Network. We expect that it sheds useful light on how the representation of Deep CNNs learned in language tasks represents the given text.

We show that **several information in the given text are not distributed across all units**. We observe AND quantify that **even a single unit** can act as a _natural language concept (e.g. morpheme, word, phrase)_ detector.

---
# Visualizing Individual Units

We align three _natural language concepts_ per unit. Most units are selectively responsive to the concepts we align. If you want to see the full results, see <TODO>.

## Natural Language Concepts

<img src="figures/natural_language/1.png" width="800px" align="center", clear="both">
<img src="figures/natural_language/2.png" width="800px" align="center", clear="both">
<img src="figures/natural_language/3.png" width="800px" align="center", clear="both">
<img src="figures/natural_language/4.png" width="800px" align="center", clear="both">
<img src="figures/natural_language/5.png" width="800px" align="center", clear="both">

## Concepts that goes beyond natural language form

We also discovered that several units tend to capture the concepts that goes beyond natural langauge form. Although it is relatively hard to _quantify_ it, we belive that futher investigation would be an interesting future direction. We visualize some units that capture abstract form concepts as follows:

### Number
<img src="figures/number/1.png" width="800px" align="center", clear="both">
<img src="figures/number/2.png" width="800px" align="center", clear="both">

### Number + Time
<img src="figures/number(time)/1.png" width="800px" align="center", clear="both">
<img src="figures/number(time)/2.png" width="800px" align="center", clear="both">

### Number + Question
<img src="figures/number+questions/1.png" width="800px" align="center", clear="both">

### Quantity
<img src="figures/quantity/1.png" width="800px" align="center", clear="both">
<img src="figures/quantity/1.png" width="800px" align="center", clear="both">

### Wh-questions
<img src="figuers/quantity/1.png" width="800px" align="center", clear="both">
<img src="figuers/quantity/2.png" width="800px" align="center", clear="both">

### A demonstrative pronoun
<img src="figuers/a_demonstrative_pronoun/1.png" width="800px" align="center", clear="both">

---
# Run
If you want to see our results without running the code, skip these parts and go to [Full visualization results](#(Optional)-Full-visualization-results)



## Prerequisites

* Python 2.7
* Tensorflow 1.10+

## Download

* Clone the code from GitHub.
```
git clone https://github.com/seilna/CNN-Units-in-NLP.git
```

* Download training data & pretrained models (~160GB space)
```
cd script
bash setup.sh 
```

* Install dependencies
```
pip install requirements.txt
```


## Running visualization of units
```
cd script
bash run.sh 
```

Visualization results are saved at `visualization/`.

## (Optional) Full visualization results
```
cd script
bash download_visualization.sh
```

---

# Reference

If you find the code useful, please cite the following paper.

```
@inproceedings{
  Na:ICLR:2019,
  title = "{Discovery of Natural Language Concepts in Individual Units of CNNs}",
  author = {Seil Na and Yo Joong Choe and Dong-Hyun Lee and Gunhee Kim},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://openreview.net/forum?id=S1EERs09YQ},
}
```

---

# Acknowledgements

We appreciate <a href="http://vision.snu.ac.kr/people/insujeon.html">Insu Jeon</a>, <a href="https://j-min.io/">Jaemin Cho</a>, <a href="https://shmsw25.github.io/">Sewon Min</a>, <a href="https://yunseokjang.github.io/">Yunseok Jang</a> and the anonymous reviewers for their helpful comments and discussions. This work was supported by <a href="https://www.kakaocorp.com/">Kakao</a> and <a href="http://kakaobrain.com/">Kakao Brain</a> corporations, IITP grant funded by the Korea government (MSIT) (No. 2017-0-01772) and Creative Pioneering Researchers Program through Seoul National University.
