import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
s=np.linspace(0,20,20)
N=20

def likelihood(domain):
    i=0
    sum=0
    while(i<N):
        sum+=pow(s[i]-domain,2)
        i=i+1
    a=-0.5*sum-(N/2)*(math.log(2*3.14))
    return a
b = np.vectorize(likelihood)
domain=np.linspace(0,20,20)

#required decoration
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0, 0, 0))
#end of decoration

plt.plot(domain,b(domain))
plt.plot(domain,b(domain),'go-')
