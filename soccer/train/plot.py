import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []

def plot_durations(show_result=False, loss_history=None):
    plt.figure(1)
    plt.clf()
    if loss_history is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, num=1, figsize=(8, 8))
    else:
        ax1 = plt.subplot(1, 1, 1)

    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    ax1.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy())

    if loss_history is not None:
        losses_t = torch.tensor(loss_history, dtype=torch.float)
        ax2.plot(losses_t.numpy())

        if len(losses_t) >= 100:
            means = losses_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax2.plot(means.numpy())

    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf()) 