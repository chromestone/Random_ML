#20 Newsgroups Ciphertext Challenge

https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge

So here is my go at this.

I first tried a simple feed forward neural network. (I tried this shortcut thing mimicking resnet but disclaimer I mixed up concatenation and add (+) woopsy)

[feedforward](feedforward.ipynb)

Second, I thought about using the inception network. I picked it because it was what came to mind when I thought "most state of the art and included in keras". I was already thinking resnet so inception_resnet_v2.

Then I realized the kaggle kernel wouldn't let me download weights...

Now I'm doing some stuff on google colab. There I ran into a problem where setting trainable affects the model. So I can't load weights into a model with different weights frozen. I could just load the previous architecture and then change it. I decided to do some inception (model-ception) with models inside models. The outer model contains a series of models frozen (pre-trained) except the last which that model's weights that I want to train.

[kaggling_20_targets_ciphertext](kaggling_20_targets_ciphertext.ipynb)

I did some training and it was bad. Worse than kernel using random forest using TF IDF. In the same notebook I started doing some crazy cosine stuff.

[kaggle_newsgroup_cipher](kaggle_newsgroup_cipher.ipynb)

I tried to visualize the inception model but the sparsity made gradient ascent look all noisy. Not at all like the cliche neural network feature increasing in abstraction images! (not included)

So maybe do away with pretrained?

I thought about triplet loss because I one hot encoded the text essentially into an image. I first started first with testing cosine proximity. I stopped early upon realizing this didn't make much sense because there are only 20 identities (newsgroups).

[kaggle_cipher_cosine](kaggle_cipher_cosine.ipynb)

Since I already built a the facenet model for 1D, let's do a softmax with nothing pretrained instead. Fingers crossed.

At first: Oh no I maybe I had set the batch size too big. Lowered from 32, the default, to 8.

Then: Oh no I used tf.losses.softmax_cross_entropy wrong! I applied a softmax activation at the end when I should not have. Oh well, retrain the 2hrs total I lost.

Maybe I shouldn't have written a custom loss? However, the architecture kind of requires it. I split the last layer into 4 where they are trained on 1 out of the 4 difficulties while other layers are shared.

All these models seem to just stop improving after a while for some reason.

I trained an additional 20 epochs on AWS. (Total of 42 epochs)

[kaggle_cipher_share_train](kaggle_cipher_share_train.ipynb)

Perhaps I could try to use a generator and train the same proportion of each difficulty every batch?

The training yielded an accuracy of 0.07441360237631875.

This is better than guessing of 0.05 but somewhat worse than the about 0.8 something accuracy acheived from a pretrained inception resnet v2 with hot encoded inputs.

Well, that's all folks.
