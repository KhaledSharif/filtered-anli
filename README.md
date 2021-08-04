## Adversarially Filtering the Adversarial NLI Dataset to Improve Accuracy

### Abstract

ANLI is a novel Natural Language Inference benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure. The dataset was collected in order to be purposefully difficult for SOTA NLI models. It therefore presents a unique opportunity to build better NLI models that better generalize to out-of-distribution inputs. This paper's main focus is on improving accuracy of NLI models on the ANLI dev and test datasets by filtering the ANLI training dataset using a novel dataset filtration technique called AFLite. 

### How to run code

The code is divided into three main parts:

1. Generating embeddings for each datapoint in the ANLI dataset using a RoBERTa-large model pretrained on MNLI (see python scripts beginning with 1)
2. Using the generated embeddings to filter the ANLI dataset using AFLite (see python scripts beginning with 2)
3. Using the filtered datasets (ANLI R1,R2,R3) to train a RoBERTa NLI model (see python scripts beginning with 3)

### References

[Adversarial NLI: A New Benchmark for Natural Language Understanding](https://github.com/facebookresearch/anli)

[Adversarial Filters of Dataset Biases](https://arxiv.org/abs/2002.04108)

### Authors

* Khaled Al-Sharif
* Nicholas Park
* Sida Chen
* Alexander Armbruster
