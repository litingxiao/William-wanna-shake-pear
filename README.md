# william-wanna-shake-pear
 This repository is for **Caltech CS155: Machine Learning and Data Mining** (2020 Winter) miniproject3. 
 
 Included are codes to build HMMs and RNNs for generating sonnets of Shakespeare's writing style by training on all 154 Shakespeare's sonnets (and Amoretti by Edmund Spenser for 2 advanced models).

## Generated sonnets:
### Sample RNN-generated sonnet:

	Then the powerful king put to the test,
	Both soul doth say thy show,
	The worst of worth the words of weather ere:
	And with the crow of well of thine to make thy large will more.
	
	Let no unkind, no fair beseechers kill,
	Though rosy lips and lovely yief.
	Her wratk doth bind the heartost simple fair,
	That eyes can see! take heed (dear heart) of this large privilege,
	
	And she with meek heart doth please all seeing,
	Or all alone, and look and moan,
	She is no woman, but senseless stone.
	But when i plead, she bids me play my part,
	
	And when i weep, in all alone,
	That he with meeks but best to be.

### Sample HMM-generated sonnet:

	The large numbers must due did be you sweet
	Love thou have black heaven love hair was stand
	Beauteous the large least in the building seat
	Child be numbers to fair fearless men brand
	
	The happy dress came now would my thy gait
	It they and that clay all she glory that
	Thou was me should tears in they art the mate
	The message and thine once make thine loud at
	
	Beseechers rack for it miss print said thou
	Sue glorious mine freshly affections growth
	Praise to have same feeds in tongue and then how
	Back seeing the black sail so my is loath
	
	Thoughts ever in in one in thine been turn
	True none thou wolf ever or she return

## Included files: 
### 1-Data-Preprocessing.ipynb
Pre-processes the Shakespearean sonnet datasets for training HMMs and RNNs, including basic pre-processing for naive HMM and naive RNN, as well as advanced pre-processing for advanced HMM and advanced RNN.

### 2-HMM-Train-Poems.ipynb
Trains a naive HMM and an advanced HMM to write Shakespearean sonnets.

HMM implementation, including Viterbi alogorithm, is in *HMM.py*, and HMM helper functions for some visualization and interpretation are in *HMM_helper.py*.

### 3-RNN-William-Wanna-Shake-Pear.ipynb
Trains a naive RNN and an advanced RNN using *keras* to write Shakespearean sonnets. (Trained on Google Colab GPUs.)

### raw_data
 Contains raw data used for training. 
 * *shakespeare.txt*: all 154 Shakespeare's sonnets.
 * *spenser.txt*: Amoretti written by Edmund Spenser in the 16th century. All 139 of Spenser’s sonnets in the Amoretti follow the same rhyme scheme and meter as Shakespeare’s sonnets.
 * *Syllable_dictionary.txt*: syllable count information in shakespeare.txt.
 * *syllable_dict_explanation.pdf*: explanation of the file Syllable_dictionary.txt
 
### processed_data
 Contains processed data after data pre-processing and HMM/RNN model training. 
 
 * Pre-processing for HMMs:
   * *{}_processed_seqs_vec.p*: pre-processed list of sequences of vectors (of words) in shakespeare.txt (and spenser.txt) used in HMMs.
   * *{}_vec2word.p*: pre-processed mapping from words to vectors used in HMMs. 
   * *{}_word2vec.p*: pre-processed inverse mapping from vectors to words used in HMMs.
   
 * Pre-processing for RNNs:
   * *{}_char_seqs_vec.p*: pre-processed list of sequences of vectors (of chars) in shakespeare.txt (and spenser.txt) used in RNNs.
   * *{}_char2vec.p*: pre-processed mapping from chars to vectors used in RNNs.
   * *{}_vec2char.p*: pre-processed inverse mapping from vectors to chars used in RNNs.
   
 * Trained models: 
   * *{}_hmm{}_A.p*: trained state transition matrix A in HMMs.
   * *{}_hmm{}_O.p*: trained observation matrix O in HMMs.
   * *basic_char_rnn.h5*: trained naive char-based RNN model.
   * *adv_char_rnn.h5*: trained advanced char-based RNN model. 
