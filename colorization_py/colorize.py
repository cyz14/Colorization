
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from math import *
from getColorExact import getColorExact, rgb2ntsc, ntsc2rgb
import gc; gc.collect()

g_name = '..\imgs\\2_grey.bmp'
c_name = '..\imgs\\2_marked.bmp'
out_name = '..\imgs\\2_res_py.bmp'

gI = ndimage.imread(g_name) / 255.0
cI = ndimage.imread(c_name) / 255.0

print gI.shape

colorIm = (np.sum(abs(gI - cI), axis=2) > 0.01)
# colorIm = colorIm * 1.0

sgI = rgb2ntsc(gI)
scI = rgb2ntsc(cI)

ntscIm = np.zeros_like(sgI)
ntscIm[:, :, 0] = sgI[:,:,0]
ntscIm[:, :, 1] = scI[:,:,1]
ntscIm[:, :, 2] = scI[:,:,2]

max_d = int( floor( log( min( ntscIm.shape[0], ntscIm.shape[1] ) ) / log(2) - 2 ) )
iu = int( floor( ntscIm.shape[0] / ( 2**(max_d-1) ) ) * ( 2**(max_d-1) ) )
ju = int( floor( ntscIm.shape[1] / ( 2**(max_d-1) ) ) * ( 2**(max_d-1) ) )
id=0; jd=0;
colorIm = colorIm[id:iu, jd:ju].copy()
ntscIm  = ntscIm[id:iu, jd:ju, :].copy()

nI = getColorExact(colorIm, ntscIm)
snI = nI
nI = ntsc2rgb(nI)
plt.imshow(nI)
plt.show()
plt.imsave(out_name, nI)