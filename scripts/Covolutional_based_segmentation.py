import torch
import torch.optim as optim
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
from skimage import segmentation
import torch.nn.init

sys.path.append(os.path.abspath('../tfg'))
from models.segmodel import SegModel

# Configuración
use_cuda = torch.cuda.is_available()
DATA_PATH = 'data/raw/'
DATA_DEST = 'data/segmentation/'
# Parámetros a tocar
MAX_ITER = 300
MIN_LABELS = 2
NCHANNEL = 100
LR = 0.1
NCONV = 2
NUM_SUPERPIXELS = 10000
COMPACTNESS = 100
VISUALIZE = 0

raw_images = os.listdir(DATA_PATH)
segmented_images = os.listdir(DATA_DEST)
images_to_segment = [image for image in raw_images if image not in segmented_images]

for image in images_to_segment:
    im = cv2.imread(DATA_PATH + image)
    im = cv2.resize(im, (1440, 1080))
    data = torch.from_numpy( np.array([im.transpose((2, 0, 1)).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(im, compactness=COMPACTNESS, n_segments=NUM_SUPERPIXELS)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model = SegModel( data.size(1), NCHANNEL, NCONV)
    if use_cuda:
        model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))
    for batch_idx in range(MAX_ITER):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, NCHANNEL )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if VISUALIZE:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imshow( image, im_target_rgb )
            cv2.waitKey(10)

        
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        if use_cuda:
            target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        print (batch_idx, '/', MAX_ITER, ':', nLabels, loss.item())

        if nLabels <= MIN_LABELS:
            print ("nLabels", nLabels, "reached minLabels", MIN_LABELS, ".")
            break

    if not VISUALIZE:
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, NCHANNEL )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        print(im_target_rgb)
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( DATA_DEST + image,im_target_rgb) 
    