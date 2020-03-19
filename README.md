# william-wanna-shake-pear
 This repo is for **Caltech CS155: Machine Learning and Data Mining** (2020 Winter) miniproject3. 
 
 Included are codes to build HMMs and RNNs for generating parody poems of Shakespeare's writing style by training on all 154 Shakespeare's sonnets.
 
### 1-Data-Preprocessing.ipynb
Pre-processes the Shakespeare's sonnet datasets for training HMMs and RNNs, including basic pre-processing for naive HMM and naive RNN, as well as advanced pre-processing for advanced HMM and advanced RNN.

### 2-HMM-Train-Poems.ipynb
Trains a naive HMM and an advanced HMM to write parody poems of Shakespearean sonnets by training on all 154 Shakespearean sonnets.

HMM implementation is contained in *HMM.py*, and HMM helper functions are in *HMM_helper.py*.

### 3-RNN-William-Wanna-Shake-Pear.ipynb
Trains a naive RNN and an advanced RNN using *keras* to write parody poems of Shakespearean sonnets by training on all 154 Shakespearean sonnets.

### raw_data
 Contains raw data used for training. 
 * *shakespeare.txt*: all 154 Shakespeare's sonnets.
 * *spenser.txt*: Amoretti written by Edmund Spenser in the 16th century. All 139 of Spenser’s sonnets in the Amoretti follow the same rhyme scheme and meter as Shakespeare’s sonnets.
 * *Syllable_dictionary.txt*: syllable count information in shakespeare.txt.
 * *syllable_dict_explanation.pdf*: explanation of the file Syllable_dictionary.txt
 
### processed_data
 Contains processed data after data pre-processing and HMM/RNN model training. 
 
 * Pre-processing for HMMs:
   * *{}_processed_seqs_vec.p*: pre-processed list of sequences of vectors (of words) in shakespeare.txt used in HMMs.
   * *{}_vec2word.p*: pre-processed mapping from words to vectors used in HMMs. 
   * *{}_word2vec.p*: pre-processed inverse mapping from vectors to words used in HMMs.
   
 * Pre-processing for RNNs:
   * *char_seqs_vec.p*: basic pre-processed list of sequences of vectors (of chars) in shakespeare.txt used in naive RNN.
   * *char2vec.p*: basic pre-processed mapping from chars to vectors used in naive RNN.
   * *vec2char.p*: basic pre-processed inverse mapping from vectors to chars used in naive RNN.
   
 * Trained models: 
   * *{}_hmm{}_A.p*: trained state transition matrix A in HMMs.
   * *{}_hmm{}_O.p*: trained observation matrix O in HMMs.
   * *char_rnn.h5*: trained naive char-based RNN model.
   * *word_rnn.h5*: trained advanced word-based RNN model. 
