import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import numpy as np


# SA visualization
# the following scripts is used to generate the animation of the search process
# if ffmpeg is not available, download ffmpeg from https://ffmpeg.org/download.html#build-windows first
# the set the window environment variable as https://www.dounaite.com/article/628e2110f8519f4c0cd4bfcb.html
# https://matplotlib.org/stable/gallery/animation/dynamic_image.html
# https://stackoverflow.com/questions/49158604/matplotlib-animation-update-title-using-artistanimation
# https://stackoverflow.com/questions/17895698/updating-the-x-axis-values-using-matplotlib-animation
def gen_search_evolution(num_tests, method_name, data, pos, gen):
    n = len(pos)
    # if num_tests == 1 and method_name == 'simulated annealing' and gen == True:
    if num_tests == 1 and gen == True:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
        plt.suptitle('Search Method : ' + method_name,
                     horizontalalignment='center',
                     verticalalignment='top',
                     fontsize='large', fontweight='normal'
                     )
        plt.subplots_adjust(wspace=0.4)  # more interval between two axes

        xlim = [0.999 * np.min(pos, 0)[0], 1.001 * np.max(pos, 0)[0]]
        ylim = [0.999 * np.min(pos, 0)[1], 1.001 * np.max(pos, 0)[1]]
        axis_font = dict(fontsize=8, fontweight='light')
        ax[0].set(xlabel='X Axis',  # for showing the tour evolution during the searching
                  ylabel='Y Axis',
                  xlim=xlim,
                  ylim=ylim,
                  xmargin=0.1,
                  ymargin=0.1
                  )
        ax[0].set_title(label='Current and Optimal Tours',
                        fontdict=axis_font
                        )

        ax[1].set(xlabel='Number of Iteration',  # for showing the search convergence
                  ylabel='Tour Length'
                  # xlim=[-1, data['count']*1.01],
                  # xticks=range(-1, data['count'], int(data['count']/10))
                  # title={'label': 'Convergence Curve of Search Method', 'fontdict': axis_font}
                  )
        ax[1].set_title(label='Convergence Curve',
                        fontdict=axis_font
                        )
        ims = []
        for i in range(len(data['sol'])):
            im = []
            sol = data['sol'][i]
            best_sol = data['best_sol'][i]
            cost = list(data['cost'])[:i]
            best_cost = list(data['best_cost'])[:i]

            if i > 2 and cost[-1] == cost[-2]:
                continue

            # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
            # lines = [[pos[sol[_]], pos[sol[(_ + 1) % n]]] for _ in range(n)]
            # line_segments = LineCollection(lines, color='b')
            # im.append(ax[0].add_collection(line_segments))
            lines = [[pos[best_sol[_]], pos[best_sol[(_ + 1) % n]]] for _ in range(n)]
            line_segments = LineCollection(lines, color='r', alpha=0.5, linewidth=2)
            im.append(ax[0].add_collection(line_segments))

            # # plot directed tour, too slow
            # # https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp
            # for j in range(n):
            #     start_pos = pos[sol[j]]
            #     end_pos = pos[sol[(j + 1) % n]]
            #     im.append(ax[0].annotate("",
            #                  xy=start_pos, xycoords='data',
            #                  xytext=end_pos, textcoords='data',
            #                  arrowprops=dict(arrowstyle='->',
            #                                  connectionstyle='arc3',
            #                                  alpha=1,
            #                                  color='b')))
            # for j in range(n):
            #     start_pos = pos[best_sol[j]]
            #     end_pos = pos[best_sol[(j + 1) % n]]
            #     im.append(ax[0].annotate("",
            #                  xy=start_pos, xycoords='data',
            #                  xytext=end_pos, textcoords='data',
            #                  arrowprops=dict(arrowstyle='->',
            #                                  connectionstyle='arc3',
            #                                  lw=2,
            #                                  alpha=0.5,
            #                                  color='r')))

            line1, = ax[1].plot(range(len(cost)), cost, color='b')
            line2, = ax[1].plot(range(len(best_cost)), best_cost, color='r')
            im.append(line1)
            im.append(line2)
            ims.append(im)

        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)
        ani.save('results/'+ method_name + '.mp4')
