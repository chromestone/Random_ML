An experiment during the fellowship and inspired by MaskTrack (https://arxiv.org/pdf/1612.02646.pdf and only kinda). The idea was to train an autoencoder to represent a 4 channel image into 3. The 4th channel would be the previous mask.

The reason for this was Mask RCNN (https://arxiv.org/abs/1703.06870 and https://github.com/matterport/Mask_RCNN) for image instance segmentation already existed but was only for a single images. What if we could encode the previous mask?

The result was pretty much expected (or at least I should have expected it). Mask RCNN just thought there was another instance! Merging a previous frame with the current caused a "double appearance".

The data comes from Workshop on Autonomous Driving (wad.ai) challenge

https://www.kaggle.com/c/cvpr-2018-autonomous-driving

## File information

[How_to_Google_Drive_in_Colab](How_to_Google_Drive_in_Colab.ipynb) - helped the team figure out porting our data saved on Google Drive

[WAD_Autoencoder_v2](WAD_Autoencoder_v2.ipynb) - the training

[AllTogetherNow](AllTogetherNow.ipynb) - combining the trained encoder with Mask RCNN. note that I trained two versions. I did not get the loss function right for the first few times until I learned normalizing was important. Regardless, both “versions” ended up producing the same effect. Interestingly, there was only one car duplicated (perhaps it was fast).
