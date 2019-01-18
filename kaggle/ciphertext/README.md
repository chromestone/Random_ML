#20 Newsgroups Ciphertext Challenge

https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge

So here is my go at this.

I first tried a simple feed forward neural network. (I tried this shortcut thing mimicking resnet but disclaimer I mixed up concatenation and add (+) woopsy)

[a relative link](feedforward.ipynb)

Second, I thought about using the inception network. I picked it because it was what came to mind when I thought "most state of the art and included in keras". I was already thinking resnet so inception_resnet_v2.

Then I realized the kaggle kernel wouldn't let me download weights...

Now I'm doing some stuff on google colab. There I ran into a problem where setting trainable affects the model. So I can't load weights into a model with different weights frozen. I could just load the previous architecture and then change it. I decided to do some inception (model-ception) with models inside models. The outer model contains a series of models frozen (pre-trained) except the last which that model's weights that I want to train.

[a relative link](kaggling_20_targets_ciphertext.ipynb)

I did some training and it was bad. Worse than kernel using random forest using TF IDF. In the same notebook I started doing some crazy cosine stuff.

[a relative link](kaggle_newsgroup_cipher.ipynb)

I tried to visualize the inception model but the sparsity made gradient ascent look all noisy. Not at all like the cliche neural network feature increasing in abstraction images! (not included)

So maybe do away with pretrained?

I thought about triplet loss because I one hot encoded the text essentially into an image. I first started first with testing cosine proximity. I stopped early upon realizing this didn't make much sense because there are only 20 identities (newsgroups).

[a relative link](kaggle_cipher_cosine.ipynb)

Since I already built a the facenet model for 1D, let's do a softmax with nothing pretrained instead. Fingers crossed.
