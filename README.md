# Visual Attention Grounding Neural Model for MMT

This code repository contains the implementation of the research project: _[A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266). It is taken from the original repository [here](https://github.com/zmykevin/A-Visual-Attention-Grounding-Neural-Model) and updated.

## 1. Overview

We introduce a novel multimodal machine translation model that utilizes parallel visual and texture information. Our model jointly optimizes learning of a shared visual-language embedding and translating languages. It does this with the aid of a visual attention grounding mechanism which links the visual semantics in the image with the corresponding textual semantics.

![An Overview of the Visual Attention NMT](https://github.com/zmykevin/A-Visual-Attention-Grounding-Neural-Model/blob/master/AGV-NMT.jpg)

## 2. Prerequisite
The code is partially tested on OpenSUSE Leap 15.2 with NVIDIA GPUs and [Conda](https://conda.io/miniconda.html) is expected as prerequisite. While we haven't tested this code with other OS system, we expect it can be run on any Linux Based OS with a minor adjustment. 

In summary, to install:
```
conda create -n torch_mmt anaconda
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda activate torch_mmt
```
You can change the CUDA version from 11.0 to the 10.1 or 10.2 as supported by your GPU drivers.
Clone this repository.
Download pre-processed Multi30K data from [here](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing).
Download METEOR paraphrase files from [here](https://github.com/cmu-mtlab/meteor/tree/master/data) and copy into the machine_translation_vision/meteor/data directory.

## 3. How to run the code?

Once you have launched the virtual environment, you can start to run our code. To train a VAG-NMT for translation from English to German, you will need to run the file nmt_multimodal_beam_DE.py (English to French, nmt_multimodal_beam_FR.py, is not fully implemented yet; only the nmt_multimodal_beam_DE.py script for training works):
```
  python nmt_multimodal_beam_DE.py --data_path path/to/data --trained_model_path path/to/save/model --sr en --tg de
```
You need to define at least four things in order to run this code: the directory for the dataset, the directory to save the trained model, the source language, and the target language. The languages that our model can work with include: English=> "en", German->"de" and French->“fr”.

To test a trained model on a test dataset, you can run `test_multimodal.py` to evaluate the trained multimodal NMT.
```
  python test_multimodal.py --data_path path/to/data --trained_model_file path/to/save/model/best_model.pt --sr en --tg de --output_path path/to/results
```
You need to define the directory for the dataset, the file containing the trained model, the source language, the target language, and the directory to save the results from the testing.

We have the Preprocessed Multi30K Dataset available in this [link](https://drive.google.com/drive/folders/1G645SexvhMsLPJhPAPBjc4FnNF7v3N6w?usp=sharing), which can be downloaded to train and test the model. One more thing, to properly use the METEOR score to evaluate the model's performance, you will need to download a set of METEOR paraphrase files and store it under the repository of machine_translation_vision/meteor/data. These paraphrase files are available to be download from [here](https://github.com/cmu-mtlab/meteor/tree/master/data).

Once the above two datasets are downloaded and placed in the appropriate folders, the example code to train and test the model is as follows.

Training:
```
python Visual-Attention-Grounding-MMT/nmt_multimodal_beam_DE.py --data_path data/Multi30K_DE --trained_model_path model_VAG-NMT_multi30k_de --batch_size 32 --eval_batch_size 16 --n_epochs 100 --eval_every 1000 --print_every 100 --save_every 10000 --sr en --tg de
```

Testing:
```
python Visual-Attention-Grounding-MMT/test_multimodal.py --data_path data/Multi30K_DE --trained_model_file model_VAG-NMT_multi30k_de/nmt_trained_imagine_model_best_BLEU.pt --batch_size 32 --eval_batch_size 16 --sr en --tg de --output_path results_model_VAG-NMT_multi30k_de_1
```

## 4. Command to run with tiny model

These commands are to verify that the code runs when we only have access to machines with a small amount of GPU memory.

Training:
```
python Visual-Attention-Grounding-MMT/nmt_multimodal_beam_DE.py --data_path data/Multi30K_DE --trained_model_path model_VAG-NMT_multi30k_de --batch_size 2048 --eval_batch_size 512 --embedding_size 4 --hidden_size 8 --shared_embedding_size 8 --n_epochs 4 --eval_every 8 --print_every 8 --save_every 8 --sr en --tg de
```

Testing:
```
python Visual-Attention-Grounding-MMT/test_multimodal.py --data_path data/Multi30K_DE --trained_model_file model_VAG-NMT_multi30k_de/nmt_trained_imagine_model_best_BLEU.pt --batch_size 2048 --eval_batch_size 512 --sr en --tg de --output_path model_VAG-NMT_multi30k_de_results_1
```

## 5. Other information

### IKEA Dataset
The collected product description multimodal machine translation benchmark crawled from IKEA website is stored under the github repo [here](https://github.com/sampalomad/IKEA-Dataset)

