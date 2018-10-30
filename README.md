#Multimodal neural pronunciation modeling for spoken languages with logographic origin

This code implements neural network models to predict pronunciation of Cantonese, using pronunciation of cognates in historically related languages (Mandarin, Vietnamese, Korean) and embeddings from logographic characters.
The code uses Keras with Tensorflow backend.
The models were presented in the paper:

Minh Nguyen, Gia H. Ngo, Nancy F. Chen,
[Multimodal neural pronunciation modeling for spoken languages with logographic origin][arXiv],
Empirical Methods in Natural Language Processing (EMNLP), 2018.

## Data
The dataset is extracted from the [UniHan database][UniHan], which is the pronunciation database of characters from Han logographic languages.

* Training set: `data/train.csv`
* Validation set: `data/validation.csv`
* Test set: `data/test.csv`

Each file consists of multiple lines, each corresponds to a logogram.
Each row consists of 4 columns, corresponding to the Unicode of the logogram, the corresponding phonemes in Mandarin, Cantonese, Korean and Vietnamese.
Examples from the training set can be shown using the following command: ``python3 preview.py -d data/train.csv``

`ids.txt` was cloned from [https://github.com/cjkvi/cjkvi-ids][ids], containing the Ideographic Description Sequence data derived from [CHISE][CHISE] project.
The Ideographic Description Sequence is used to construct logoraphs' embedding.

## Reproducing the paper results

1. Clone this repository.  
   ```sh
   git clone https://github.com/nguyen-binh-minh/logographic  
   cd logographic
   ```

2. Install [Anaconda](https://www.anaconda.com/)

3. Set up Python environment with Anaconda  
   ```sh
   conda env create --name py3_env --file environment.yaml
   ```

4. Replicate the experiments  
   ```sh
   source activate py3_env  
   ./scripts/example_mlp_bor.sh  # MLP with bag-of-radicals input  
   ./scripts/example_lstm_geod.sh  # LSTM with GeoD input  
   ./scripts/example_mlp_bor_ph.sh  # MLP with bag-of-radicals input and cognates' phonemes input  
   ./scripts/example_lstm_geod_ph.sh  # LSTM with GeoD input and cognates' phonemes input
   ```

## License

The code in this repository is released under the terms of the [MIT license](LICENSE.txt).  
The Ideographic Description Sequence data is under [GPLv2 license](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

[arXiv]:  https://arxiv.org/abs/1809.04203
[UniHan]: https://www.unicode.org/charts/unihan.html
[CHISE]:  http://www.chise.org
[ids]:    https://github.com/cjkvi/cjkvi-ids
