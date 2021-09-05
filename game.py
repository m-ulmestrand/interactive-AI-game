import numpy as np
from matplotlib import pyplot as plt
from movement_handling import move, confine_particles
from prey.agent import Prey
import keyboard
from matplotlib.font_manager import FontProperties


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 12})

it = FontProperties()
it.set_family('serif')
it.set_name('Times New Roman')
it.set_style('italic')
it.set_size(16)

times = FontProperties()
times.set_family('serif')
times.set_name('Times New Roman')
times.set_size(16)


box_size = 100
radius = 5
speed = 0.5
prey_radius = 0.5
n_prey = 100
indices = np.arange(n_prey)
position = (np.array((box_size, box_size)) / 2).reshape((1, 2))
v = np.array((0, speed), dtype=float).reshape((1, 2))
angle = np.pi / 2
angle_diff = 0.07
period = np.pi * 2
kills_history = np.zeros((0, 2))
loss_history = np.zeros((0, 2))

prey = Prey(box_size, n_prey, buffer_size=5000, buffer_behaviour="discard_old", epsilon_final=0.15, episode_length=800)
colormap = plt.cm.get_cmap('viridis')
predator_color = np.array([0.5, 0.1, 0.2, 1])
colors = np.array([colormap(0.2 + 0.6 * i/(n_prey-1)) for i in range(n_prey)] + [predator_color])

fig, ax = plt.subplots()
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect('equal', adjustable='box')

generation = 1
s = np.array([10*prey.radius**2] * n_prey + [20*radius**2])
scatter_plot = ax.scatter([None] * (1+n_prey), [None] * (1+n_prey), s=s, facecolors=colors, edgecolors='k')
fig.suptitle(r"Generation ${}$: $\epsilon = {}$   Previous generation kills: ${}$".format(
    generation, prey.get_epsilon(), 0))
fig.canvas.draw()
plt.show(block=False)


def keyboard_event():
    global angle
    if keyboard.is_pressed('a') or keyboard.is_pressed('left arrow'):
        angle += angle_diff
        v[0, 0] = np.cos(angle) * speed
        v[0, 1] = np.sin(angle) * speed
    elif keyboard.is_pressed('d') or keyboard.is_pressed('right arrow'):
        angle -= angle_diff
        v[0, 0] = np.cos(angle) * speed
        v[0, 1] = np.sin(angle) * speed
    if abs(angle) > period:
        angle -= np.sign(angle) * period


while plt.fignum_exists(fig.number):
    keyboard_event()

    confine_particles(position, v, box_size, box_size, radius)
    move(position, v)

    prey.select_actions(position[0], radius, v[0])
    dead_indices = prey.dead_agents[:, 0]
    reinitialize = prey.reward()
    reinforced_loss = prey.reinforce(train_size=500)

    if reinforced_loss is not None:
        t = generation + prey.count / prey.episode_length
        loss_history = np.append(loss_history, np.array([[t, reinforced_loss]]), axis=0)

    if reinitialize is not None:
        kills_history = np.append(kills_history, np.array([[generation, dead_indices.size]]), axis=0)
        generation += 1
        fig.suptitle(r"Generation ${}$: $\epsilon = {}$   Previous generation kills: ${}$".format(
            generation, round(prey.get_epsilon(), 2), dead_indices.size))
        colors = np.array([colormap(0.2 + 0.6 * i / (n_prey - 1)) for i in range(n_prey)] + [predator_color])

    inds = indices[~np.isin(indices, dead_indices)]
    all_inds = np.append(inds, n_prey)
    scatter_plot.set_offsets(np.append(prey.positions[inds], position, axis=0))
    scatter_plot.set_facecolor(colors[all_inds])
    scatter_plot.set_sizes(s[all_inds])
    fig.canvas.draw()
    fig.canvas.flush_events()

stats_figure, stats_ax = plt.subplots(1, 2, sharex=True)
stats_figure.set_size_inches(12.5, 5)
stats_ax[0].plot(kills_history[:, 0], kills_history[:, 1], 'k')
stats_ax[0].set_title("Number of kills per generation")
stats_ax[0].set_ylabel("Kills")
stats_ax[1].plot(loss_history[:, 0], loss_history[:, 1], 'k')
stats_ax[1].set_title("Loss per generation")
stats_ax[1].set_ylabel("Loss")
stats_figure.text(0.5, 0.02, 'Generation', ha='center')
plt.show()
