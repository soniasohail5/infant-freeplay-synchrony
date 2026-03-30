#!/bin/bash

REMOTE="infantresearch@2MC9W54:/C:/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
LOCAL="../data"

echo "Syncing dataset..."
rsync -avz $REMOTE $LOCAL

echo "Done."
