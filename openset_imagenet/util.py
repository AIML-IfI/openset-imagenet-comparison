"""Set of utility functions to produce evaluation figures and histograms."""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import LogLocator
import matplotlib.cm as cm
import yaml


class NameSpace:
    def __init__(self, config):
        # recurse through config
        config = {name: NameSpace(value) if isinstance(value, dict) else value
                  for name, value in config.items()}
        self.__dict__.update(config)

    def __repr__(self):
        return "\n".join(k+": " + str(v) for k, v in vars(self).items())

    def dump(self, indent=4):
        return yaml.dump(self.dict(), indent=indent)

    def dict(self):
        return {k: v.dict() if isinstance(v, NameSpace) else v for k, v in vars(self).items()}


def load_yaml(yaml_file):
    """Loads a YAML file into a nested namespace object"""
    config = yaml.safe_load(open(yaml_file, 'r'))
    return NameSpace(config)


def dataset_info(protocol_data_dir):
    """ Produces data frame with basic info about the dataset. The data dir must contain train.csv,
    validation.csv and test.csv, that list the samples for each split.
    Args:
        protocol_data_dir: Data directory.
    Returns:
        data frame: contains the basic information of the dataset.
    """
    data_dir = Path(protocol_data_dir)
    files = {'train': data_dir / 'train.csv', 'val': data_dir / 'validation.csv',
             'test': data_dir / 'test.csv'}
    pd.options.display.float_format = '{:.1f}%'.format
    data = []
    for split, path in files.items():
        df = pd.read_csv(path, header=None)
        size = len(df)
        kn_size = (df[1] >= 0).sum()
        kn_ratio = 100 * kn_size / len(df)
        kn_unk_size = (df[1] == -1).sum()
        kn_unk_ratio = 100 * kn_unk_size / len(df)
        unk_unk_size = (df[1] == -2).sum()
        unk_unk_ratio = 100 * unk_unk_size / len(df)
        num_classes = len(df[1].unique())
        row = (split, num_classes, size, kn_size, kn_ratio, kn_unk_size,
               kn_unk_ratio, unk_unk_size, unk_unk_ratio)
        data.append(row)
    info = pd.DataFrame(data, columns=['split', 'classes', 'size', 'kn size', 'kn (%)',
                                       'kn_unk size', 'kn_unk (%)', 'unk_unk size', 'unk_unk (%)'])
    return info


def read_array_list(file_names):
    """ Loads npz saved arrays
    Args:
        file_names: dictionary or list of arrays
    Returns:
        Dictionary of arrays containing logits, scores, target label and features norms.
    """
    list_paths = file_names
    arrays = defaultdict(dict)

    if isinstance(file_names, dict):
        for key, file in file_names.items():
            arrays[key] = np.load(file)
    else:
        for file in list_paths:
            file = str(file)
            name = file.split('/')[-1][:-8]
            arrays[name] = np.load(file)
    return arrays


def calculate_oscr(gt, scores, unk_label=-1):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes] or [N_samples, N_classes+1]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns.
            Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    pred_class = np.argmax(scores, axis=1)
    max_score = np.max(scores, axis=1)
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    for tau in np.unique(target_score)[:-1]:
        val = ((pred_class[kn] == gt[kn]) & (target_score > tau)).sum() / total_kn
        ccr.append(val)

        val = (unk & (max_score > tau)).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def plot_single_oscr(x, y, ax, exp_name, color, baseline, scale):
    line_style = 'solid'
    line_width = 1
    if baseline:  # The baseline is always the first array
        line_style = 'dashed'
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Manual limits
        ax.set_ylim(0.09, 1)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        loc_min = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(loc_min)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif scale == 'semilog':
        ax.set_xscale('log')
        # Manual limits
        ax.set_ylim(0.0, 0.8)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        loc_min = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(loc_min)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_ylim(0.0, 0.8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # , prune='lower'))
    # Remove fpr=0 since it cause errors with different ccrs and logarithmic scale.
    if len(x):
        non_zero = x != 0
        x = x[non_zero]
        y = y[non_zero]
    ax.plot(x, y, label=exp_name, linestyle=line_style, color=color, linewidth=line_width)
    return ax


def plot_oscr(arrays, methods, scale='linear', title=None, ax_label_font=13,
              ax=None, unk_label=-1,):

    color_palette = cm.get_cmap('tab10', 10).colors

    assert len(arrays) == len(methods)

    for idx, array in enumerate(arrays):
        has_bg = methods[idx] == "garbage"

        if array is None:
            ccr, fpr = [], []
        else:
            gt = array['gt']
            scores = array['scores']

            if has_bg:    # If the loss is BGsoftmax then removes the background class
                scores = scores[:, :-1]
            ccr, fpr = calculate_oscr(gt, scores, unk_label)

        ax = plot_single_oscr(x=fpr, y=ccr,
                              ax=ax, exp_name=methods[idx],
                              color=color_palette[idx], baseline=False,
                              scale=scale)
    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)
    return ax


def get_histogram(array, unk_label=-1,
                  metric='score',
                  bins=100,
                  drop_bg=False,
                  log_space=False,
                  geom_space_limits=(1, 1e2)):
    """Calculates histograms of scores or norms"""
    score = array['scores']
    kn_metric, unk_metric = None, None
    if drop_bg:  # if background class drop the last column of scores
        score = score[:, :-1]
    gt = array['gt'].astype(np.int64)
    features = array['features']
    norms = np.linalg.norm(features, axis=1)
    kn = (gt >= 0)
    unk = gt == unk_label
    if metric == 'score':
        kn_metric = score[kn, gt[kn]]
        unk_metric = np.amax(score[unk], axis=1)
    elif metric == 'norm':
        kn_metric = norms[kn]
        unk_metric = norms[unk]
    if log_space:
        lower, upper = geom_space_limits
        bins = np.geomspace(lower, upper, num=bins)
    kn_hist, kn_edges = np.histogram(kn_metric, bins=bins)
    unk_hist, unk_edges = np.histogram(unk_metric, bins=bins)
    return kn_hist, kn_edges, unk_hist, unk_edges


def get_best_arrays(files_dict):
    best_paths = dict()
    for name, path in files_dict.items():
        exception_list = ['$S_2$', '$E_2$', '$O_2$', '$S_3$', '$E_3$', '$O_3$',
                          '$S_1$', '$E_1$', '$O_1$']
        if name in exception_list:
            best_paths[name] = path
        best_paths[name] = Path(str(path).replace('_curr_', '_best_'))
    return best_paths
