#!/bin/bash

# download MS COCO dataset
#
# ref: http://mscoco.org/dataset/#download

echo "Downloading..."

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip && \

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip && \
    
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip && \

wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip && \

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip && \
    
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip && \
    
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip && \

wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip && \

echo "Done."