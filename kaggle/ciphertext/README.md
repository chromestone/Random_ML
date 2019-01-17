#20 Newsgroups Ciphertext Challenge

https://www.kaggle.com/c/20-newsgroups-ciphertext-challenge

So here's my go at this.

I first tried a simple feed forward neural network. (I tried this shortcut thing mimicking resnet but disclaimer I mixed up concatenation and add (+) woopsy)

Second, I thought about using the inception network. I picked it because it was what came to mind when I thought "most state of the art and included in keras". I was already thinking resnet so inception_resnet_v2.

Then I realized the kaggle kernel wouldn't let me download weights...

Now I'm doing some stuff on google colab. There I ran into a problem where setting trainable affects the model. So I can't load weights into a model with different weights frozen. I could just load the previous architecture and then change it. I decided to do some inception (model-ception) with models inside models. The outer model contains a series of models frozen (pre-trained) except the last which that model's weights that I want to train.

I tried to visualize the inception model but the sparsity made gradient ascent look all noisy. Not at all like the cliche neural network feature increasing in abstraction images! So maybe do away with pretrained?

I thought about triplet loss and then started first with testing cosine proximity. I stopped early upon realizeing this didn't make much sense because there are only 20 identities (newsgroups).

The competition ends.

I come back and see how the facenet model with softmax and nothing pretrained works on this dataset.
