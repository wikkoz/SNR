from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import numpy as np

image = ndimage.imread('/home/wiktor/projects/studia/SNR/fruits-360/Training/Kiwi/0_100.jpg')
fft2 = fftpack.fft2(image[:,:,0])
print(image[:,:,2].shape)
plt.imshow(np.log(abs(fft2)))
plt.show()
