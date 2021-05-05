"""
Charles Walker 
CPSC 322
Section 02
PA2 
"""
import mysklearn.myutils as utils
import matplotlib.pyplot as plt

def bar_chart(x, y, col_name):
    plt.figure()
    fig, ax = plt.subplots()
    plt.bar(x, y)
    plt.xlabel(col_name)
    plt.ylabel('count')
    ax.set_xticklabels(x, rotation=90)
    plt.show()
    
def pie_chart_example(x, y):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()
    
def freq_diagram1(mypy, col_name, cutoffs):
    utils.conv_num(mypy)
    values = utils.get_col(mypy, col_name)
    values.sort()
    freqs = utils.compute_bin_frequencies(values, cutoffs)
    plt.figure()
    
    plt.bar(cutoffs[:-1], freqs, width=cutoffs[1] - cutoffs[0], align="edge")
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.show()

def freq_diagram2(mypy, col_name, bins):
    utils.conv_num(mypy)
    values = utils.get_col(mypy, col_name)
    values.sort()
    cutoffs = utils.compute_equal_width_cutoffs(values, bins)
    freqs = utils.compute_bin_frequencies(values, cutoffs)
    plt.figure()
    print(cutoffs)
    
    plt.bar(cutoffs[:-1], freqs, width=cutoffs[1] - cutoffs[0], align="edge")
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.show()
    
def histo(mypy, col_name):
    # data is a 1D list of data values
    utils.conv_num(mypy)
    data = utils.get_col(mypy, col_name)
    plt.figure()
    plt.hist(data, bins=10) # default is 10
    plt.xlabel(col_name)
    plt.show()
    
def scatter(mypy, col1, col2):
    utils.conv_num(mypy)
    y = utils.get_col(mypy, col2)
    x = utils.get_col(mypy, col1)

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(col1)
    plt.ylabel(col2)
#     plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5); 
    plt.show()
    
def trendline(mypy, col1, col2):
    utils.conv_num(mypy)
    y = utils.get_col(mypy, col2)
    x = utils.get_col(mypy, col1)

    m, b = utils.compute_slope_intercept(x, y)
    r = utils.compute_corr_coef(x, y)
    cov = utils.compute_covar(x, y)
    plt.figure()
    plt.xlabel(col1)
    plt.ylabel(col2)
    string = f"Coef: {r} Cov: {cov}"
    plt.suptitle(string, fontsize=10)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5); 
    plt.show()
    
def pie_chart(x, y):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show() 
    

def box_plot(distributions, labels):

    plt.figure()
    plt.boxplot(distributions)

    plt.xticks(list(range(1, len(labels) + 1)), labels)

    plt.annotate("$\mu=100$", xy=(1.5, 100), xycoords="data", horizontalalignment="center")
    plt.annotate("$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction", 
                 horizontalalignment="center", color="blue")
    plt.show()
