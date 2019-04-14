from numpy import random
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * random.randn(greyhounds)
lab_height = 28 + 4 * random.randn(labs)


plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()
