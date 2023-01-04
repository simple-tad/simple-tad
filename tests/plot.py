import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

x = np.linspace(-100, 100, 200)

# mu = -0.73353
# variance = 16.0974 
# sigma = math.sqrt(variance)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))

mu = 50
variance = 1
sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

# mu = 6.79622
# variance = 229.902
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

# -0.73353 16.0974 
# 0 0 
# 6.79622 229.902 