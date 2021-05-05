"""
Charles Walker 
CPSC 322
Section 02
PA2 
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
""""


# TODO: your reusable plotting functions here
import matplotlib.pyplot as plt
import utils

def bar_chart(x, x_labels, y, y_title, x_title):
    """
    """
    # Box Plot 
    plt.figure()
    plt.xticks(x, x_labels, rotation=90)
    plt.title(x_title + " " + y_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.bar(x, y)
    plt.show()

def pie_chart(x_labels, y, title):
    """
    """
    plt.figure()
    plt.title(title)
    plt.pie(y, labels= x_labels, autopct="%1.1f%%")
    plt.show()

def frequency_diagram(cutoffs, bin_freqs, y_title, x_title):
    """
    """
    width = round(abs(cutoffs[1] - cutoffs[0]))

    labels = []
    label_locations = []
    for i in range(len(cutoffs)):
        if i > 0:
            labels.append(str(cutoffs[i - 1]) + "-" + str(cutoffs[i]))
            label_locations.append(cutoffs[i - 1])

    plt.title(x_title + " " + y_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xticks(label_locations, labels, rotation=90)

    for j in range(len(bin_freqs)):
        plt.bar(cutoffs[j], bin_freqs[j], width, color='#581845')

def histogram(title, column_Category):
    """
    """
    plt.figure()
    plt.title(title + " Histogram")
    plt.hist(column_Category)
    plt.show()

def histogram_double(title, column_list1, column_list2):
    """
    """
    plt.figure()
    plt.title(title + " Histogram")
    plt.hist(column_list1, bins=10)
    plt.hist(column_list2, bins=20)
    plt.show()

def scatter_with_linear_regr(x, y, m, b, regression, covariance, y_label, x_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs " + y_label + ", r = " + str(regression) + ", cov = " + str(covariance))
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    plt.show()

def multi_box_wisker(code, tick_names, amt, all_genres1, all_genres2, title):
    plt.figure()
    exec(code)
    plt.title(title)
    plt.xticks(amt, tick_names, rotation=90)
    plt.show()