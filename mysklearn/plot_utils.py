import numpy as np
import matplotlib.pyplot as plt

def boxplot_by_label(table, feature_name, label_name):
    feat = table.get_column(feature_name, False)
    labs = table.get_column(label_name, False)

    groups = {}
    n = len(feat)
    for i in range(n):
        v = feat[i]
        lab = labs[i]
        try:
            fv = float(v)
        except Exception:
            continue
        if lab not in groups:
            groups[lab] = []
        groups[lab].append(fv)

    ordered_labels = sorted(groups.keys())
    data = [groups[lab] for lab in ordered_labels]

    categorical_continuous_boxplot(
        title=f"{feature_name} by {label_name}",
        data=data,
        labels=ordered_labels,
        xlabel=label_name,
        ylabel=feature_name
    )

def plot_frequency_diagrams(title, counts, xlabel, ylabel='Count'):
    plt.figure(figsize=(8, 5))
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_histogram(title, column_data, num_bins, xlabel, ylabel = 'Count'):
    plt.figure(figsize=(8, 5))
    plt.hist(column_data, bins = num_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatterplot_with_label(title, x_column_data, y_column_data, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_column_data, y_column_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def categorical_continuous_boxplot(title, data, labels, xlabel, ylabel): 
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels = labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_correlation_heatmap(table, col_names=None, figsize=(10, 8)):
    if col_names is None:
        numeric_cols = []
        for name in table.column_names:
            col = table.get_column(name, include_missing_values=False)
            ok = True
            for v in col:
                try:
                    float(v)
                except Exception:
                    ok = False
                    break
            if ok:
                numeric_cols.append(name)
        col_names = numeric_cols

    X = []
    for row in table.data:
        new_row = []
        for name in col_names:
            idx = table.column_names.index(name)
            val = row[idx]
            try:
                new_row.append(float(val))
            except Exception:
                new_row.append(np.nan)
        X.append(new_row)
    X = np.array(X, dtype=float)

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    corr = np.corrcoef(X, rowvar=False)

    plt.figure(figsize=figsize)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    n = len(col_names)
    plt.xticks(range(n), col_names, rotation=90)
    plt.yticks(range(n), col_names)

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
