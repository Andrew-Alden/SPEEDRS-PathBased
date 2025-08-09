import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path
from collections import defaultdict


def compute_ci(data_dict, x_range, alpha_ci=0.95):

    """
    Compute symmetric confidence interval.
    :param data_dict: Dictionary containing data points. Keys correspond to items in x_range and the values of the
                      dictionary are the model outputs corresponding to the current key.
    :param x_range: List of model inputs.
    :param alpha_ci: Confidence interval percentage. Default is 0.95.
    :return: List of confidence interval, one confidence interval per value in x_range.
    """

    ci = []
    p_lower = ((1.0 - alpha_ci) / 2.0) * 100
    p_upper = (alpha_ci + ((1.0 - alpha_ci) / 2.0)) * 100

    for key in x_range:
        ci.append([np.nanpercentile(data_dict[key], p_lower), np.nanpercentile(data_dict[key], p_upper)])
    return ci


def construct_plot(ax, targets, pred_dict, x_range, alphas, colours, label, linewidths, x_label, y_label, target_label,
                   filename=None, alpha_ci=0.95, title='MSE', error_fn=None):

    """
    Construct scatter plots with confidence intervals. 
    :param ax: Axis on which to plot the data. 
    :param targets: Target values. 
    :param pred_dict: Dictionary containing model outputs. Keys correspond to items in x_range and the values of the
                      dictionary are the model outputs corresponding to the current key. 
    :param x_range: List of model inputs.
    :param alphas: List of alpha values used for plotting. 
    :param colours: List of colours.
    :param label: Model output label.
    :param linewidths: List of line widths.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param target_label: Target label.
    :param filename: Filename for the plot. If filename is None the plot is not saved. Default is None.
    :param alpha_ci: Confidence interval percentage. Default is 0.95.
    :param title: Plot title. If title contains MSE, then the MSE is displayed in the title. If error_fn is not None,
                  then the evaluation based on this error function is included in the title. Default is 'MSE'.
    :param error_fn: Error function. Default is None.
    :return: Nothing.
    """

    ax.plot(x_range, [np.nanmean(pred_dict[key]) for key in x_range], label=label, color=colours[0],
            alpha=alphas[0], linewidth=linewidths[0])

    ci = compute_ci(pred_dict, x_range, alpha_ci=alpha_ci)
    ax.fill_between(x_range, np.asarray(ci)[:, 0], np.asarray(ci)[:, 1], color=colours[1], alpha=alphas[1])

    if type(targets) == type({}) or type(targets) == type(defaultdict(list)):
        ax.scatter(x_range, [np.nanmean(targets[key]) for key in x_range], label=target_label, color=colours[2],
                   alpha=alphas[2], marker='o', linewidth=linewidths[1])
    else:
        ax.scatter(x_range, targets, label=target_label, color=colours[2],
                   alpha=alphas[2], marker='o', linewidth=linewidths[1])
    ax.grid(True, color='black', alpha=0.2, linestyle='--')
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.legend(loc='upper left', fontsize=15)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    if title is not None:
        if type(targets) == type({}) or type(targets) == type(defaultdict(list)):
            error_targets = [np.nanmean(targets[key]) for key in x_range]
        else:
            error_targets = targets
        if title.lower() == 'mse':
            title = f'MSE: {np.round(mean_squared_error([np.nanmean(pred_dict[key]) for key in x_range], error_targets), 6)}'
        elif 'mse' in title.lower():
            title += f': {np.round(mean_squared_error([np.nanmean(pred_dict[key]) for key in x_range], error_targets), 6)}'
        else:
            if error_fn is not None:
                title += f'{np.round(error_fn([np.nanmean(pred_dict[key]) for key in x_range], error_targets), 6)}'
            else:
                pass
        ax.set_title(title, fontsize=15)

    if filename is not None:
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(f'{filename}', bbox_inches='tight', format='svg', dpi=1200)