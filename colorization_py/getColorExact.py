# encoding=utf-8
'''
Description
'''
import numpy as np
from scipy import sparse
from math import log

def getColorExact(colorIm, ntscIm):
    ''' Colorize using optimization,
        Project homepage is http://www.cs.huji.ac.il/~yweiss/Colorization/
    '''
    nI = np.zeros_like(ntscIm)
    nI[:, :, 0] = ntscIm[:, :, 0]
    m = ntscIm.shape[0]
    n = ntscIm.shape[1]
    img_size = m * n
    img_mat = np.arange(img_size)
    lbl_idxs = img_mat.reshape(m, n)[np.where(colorIm)]
    lbl_idxs.sort()
    img_mat = img_mat.reshape(m, n)

    wd = 1 # width of window
    length = 0 # total length
    consts_len = 0
    col_inds = np.zeros(img_size * (2*wd+1)**2)
    row_inds = np.zeros(img_size * (2*wd+1)**2)
    vals     = np.zeros(img_size * (2*wd+1)**2)
    gvals    = np.zeros(img_size * (2*wd+1)**2)
    
    for i in xrange(m):
        for j in xrange(n):
            if not colorIm[i, j]:
                tlen = 0
                for ii in range(max(0,i-wd), min(i+wd+1,m-1)):
                    for jj in range(max(0,j-wd), min(j+wd+1,n-1)):
                        if (ii != i) or (jj != j):
                            gvals[tlen] = ntscIm[ii,jj,0] # Y in YUV
                            row_inds[length] = consts_len
                            col_inds[length] = img_mat[ii, jj]
                            tlen += 1
                            length += 1

                t_val = ntscIm[i, j, 0]
                gvals[tlen] = t_val
                # calculate variance, including t_val
                cvar = np.mean( (gvals[0:tlen+1]- np.mean(gvals[0:tlen+1]))**2 )
                csig = cvar * 0.6
                # calculate min sqr dist, not considering t_val itself
                mgv = min( (gvals[0:tlen] - t_val)**2 )
                if csig < (-mgv/log(0.01)):
                    csig = -mgv/log(0.01)
                if csig < 2e-6:
                    csig = 2e-6
                
                # calculate weight_rs between, r != s, so no gvals[tlen+1]
                gvals[0:tlen] = np.exp( -(gvals[0:tlen] - t_val)**2 / csig )
                gvals[0:tlen] = gvals[0:tlen] / np.sum(gvals[0:tlen])
                vals[length-tlen:length] = -gvals[0:tlen]
            # end if not colorIm[i, j]
            
            row_inds[length] = consts_len
            col_inds[length] = img_mat[i, j]
            vals[length] = 1
            length += 1
            consts_len = consts_len + 1
        # end for j
    # end for i

    vals     = vals[0:length]
    row_inds = row_inds[0:length]
    col_inds = col_inds[0:length]

    A = sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(consts_len, img_size)).tocsr()
    b = np.zeros(A.shape[0])

    for t in range(1,3):
        curIm = ntscIm[:, :, t]
        b[lbl_idxs] = curIm.reshape(img_size)[lbl_idxs]
        new_vals = sparse.linalg.spsolve(A,  b)
        nI[:, :, t] = np.reshape(new_vals, (m, n))
    # end for

    return nI


def rgb2ntsc(A):
    ' Convert RGB to YIQ '
    YIQ = np.zeros_like(A)
    YIQ[:, :, 0] = 0.299 * A[:,:,0] + 0.587 * A[:,:,1] + 0.114 * A[:,:,2]
    YIQ[:, :, 1] = 0.596 * A[:,:,0] - 0.275 * A[:,:,1] - 0.321 * A[:,:,2]
    YIQ[:, :, 2] = 0.212 * A[:,:,0] - 0.523 * A[:,:,1] + 0.311 * A[:,:,2]
    return YIQ


def ntsc2rgb(A):
    ' Convert YIQ to RGB '
    RGB = np.zeros_like(A)
    RGB[:, :, 0] = 1.000 * A[:,:,0] + 0.956 * A[:,:,1] + 0.621 * A[:,:,2]
    RGB[:, :, 1] = 1.000 * A[:,:,0] - 0.272 * A[:,:,1] - 0.647 * A[:,:,2]
    RGB[:, :, 2] = 1.000 * A[:,:,0] - 1.106 * A[:,:,1] + 1.703 * A[:,:,2]
    RGB = np.where(RGB > 0, RGB, 0)
    RGB = np.where(RGB <= 1., RGB, 1)
    return RGB
