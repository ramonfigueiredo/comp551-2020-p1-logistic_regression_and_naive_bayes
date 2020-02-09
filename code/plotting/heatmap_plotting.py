import matplotlib.pyplot as plt
import seaborn as sns


def heatmap_plotting(dataset, print_correlation_matrix=False, plot_heatmap_values=False, show_plotting=False, save_plotting=False, plotting_path='heatmap.png'):
    sns.set(style="white")

    # Compute the correlation matrix
    corr = dataset.corr()
    if print_correlation_matrix:
        print('Correlation matrix')

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    heatmap_plot = sns.heatmap(corr, xticklabels=True, yticklabels=True, annot=plot_heatmap_values, cmap=cmap,
                               vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if show_plotting:
        plt.show()
    if save_plotting:
        fig = heatmap_plot.get_figure()
        fig.savefig(plotting_path)
