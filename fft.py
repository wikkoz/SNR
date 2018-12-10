from scipy import fftpack, ndimage
import matplotlib.pyplot as plt

image = ndimage.imread('fruits-360/Training/Kiwi/0_100.jpg')
fft2 = fftpack.fft2(image)

plt.imshow(abs(fft2))
plt.show()