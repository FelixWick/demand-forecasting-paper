import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1)

x = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), 10000)
axs[0].fill_between(x, norm.pdf(x))
axs[0].set_title('f(x)')
axs[0].axes.xaxis.set_ticklabels([])
axs[1].plot(x, norm.cdf(x), 'k--')
axs[1].set_title('F(x)')

fig.savefig('PdfCdf.pdf')

