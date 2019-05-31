import numpy as np

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.
  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.
  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.
  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  N,C,H,W = x.shape
  F,_,HH,WW = w.shape
  S = conv_param['stride']
  P = conv_param['pad']
  Ho = int(1 + (H + 2 * P - HH) / S)
  Wo = int(1 + (W + 2 * P - WW) / S)
  x_pad = np.zeros((N,C,H+2*P,W+2*P))
  x_pad[:,:,P:P+H,P:P+W]=x
  #x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
  out = np.zeros((N,F,Ho,Wo))
 
  for f in range(F):
    for i in range(Ho):
      for j in range(Wo):
        # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
        out[:,f,i,j] = np.sum(x_pad[:, :, i*S : i*S+HH, j*S : j*S+WW] * w[f, :, :, :], axis=(1, 2, 3))
        #print(out)
    out[:,f,:,:]+=b[f]
  cache = (x, w, b, conv_param)
  return out, cache

  
  
  
  
  
'''
x_shape = (2, 3, 4, 4) #n,c,h,w
w_shape = (2, 3, 3, 3) #f,c,hw,ww
x = np.ones(x_shape)
w = np.ones(w_shape)
b = np.array([1,2])
 
conv_param = {'stride': 1, 'pad': 0}
out, _ = conv_forward_naive(x, w, b, conv_param)
 
print(out)
print(out.shape)  #n,f,ho,wo
'''
def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.
  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
 
  N, F, H1, W1 = dout.shape
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  HH = w.shape[2]
  WW = w.shape[3]
  S = conv_param['stride']
  P = conv_param['pad']
 
 
  dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
  x_pad = np.pad(x, [(0,0), (0,0), (P,P), (P,P)], 'constant')
  dx_pad = np.pad(dx, [(0,0), (0,0), (P,P), (P,P)], 'constant')
  db = np.sum(dout, axis=(0,2,3))
 
  for n in range(N):
    for i in range(H1):
      for j in range(W1):
        # Window we want to apply the respective f th filter over (C, HH, WW)
        x_window = x_pad[n, :, i * S : i * S + HH, j * S : j * S + WW]
 
        for f in range(F):
          dw[f] += x_window * dout[n, f, i, j] #F,C,HH,WW
          #C,HH,WW
          dx_pad[n, :, i * S : i * S + HH, j * S : j * S + WW] += w[f] * dout[n, f, i, j]
 
  dx = dx_pad[:, :, P:P+H, P:P+W]
 
  return dx, dw, db
  
def backward(residual, cache):
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    # db = np.zeros_like(self.b)

    db = np.sum(residual, axis=(0, 2, 3))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):  # compute dw
                dw[k, :, :, :] += np.sum(x_pad_masked * (residual[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):  # compute dx_pad
                temp_w = np.rot90(w,2,(2,3))#这种写法不旋转
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((temp_w[:, :, :, :] * (residual[n, :, i,j])[:, None, None, None]), axis=0)
    dx[:,:,:,:] = dx_pad[:, :, pad:pad+H, pad:pad+W]
    '''
    self.w -= self.lr * (dw + self.prev_gradient_w * self.reg)
    self.b -= self.lr * db
    self.prev_gradient_w = self.w
    '''
    return dx
    
  
x_shape = (2, 3, 4, 4)
w_shape = (2, 3, 3, 3)
x = np.ones(x_shape)
w = np.ones(w_shape)
b = np.array([1,2])
 
conv_param = {'stride': 1, 'pad': 0}
 
Ho = int((x_shape[3]+2*conv_param['pad']-w_shape[3])/conv_param['stride']+1)
Wo = Ho
 
dout = np.ones((x_shape[0], w_shape[0], Ho, Wo))

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)
print(dx)
dxx = backward(dout, cache)
print(dxx)
'''
print("out shape",out.shape)
print("dw==========================")
print(dw)
print("dx==========================")
print(dx)
print("db==========================")
print(db)
'''
