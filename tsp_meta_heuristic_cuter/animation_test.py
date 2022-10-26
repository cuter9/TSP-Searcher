import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from IPython.display import HTML

# Creating Data
x = np.linspace(0, 10, 100)


# defining a function to return sine of input values.
def fun(i):
    y = np.sin(i)
    return y


x_y = fun(x)

# Creating matplotlib figure and camera object
fig = plt.figure()
plt.xlim(0, 10)
plt.ylim(-2, 2)
camera = Camera(fig)

# Looping the data and capturing frame at each iteration
for i in x:
    plt.plot(x, x_y, color='green', lw=0.8)
    f = plt.scatter(i, fun(i), color='red', s=200)
    plt.title('tracing a sin function')
    camera.snap()


# Creating the animation from captured frames
animation = camera.animate(interval=200, repeat=True, repeat_delay=500)
# HTML(animation.to_html5_video()) # for Viewing in Jupyter Notebooks
animation.save('results/sine_wave.mp4')
