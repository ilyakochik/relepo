import logging
import matplotlib.pyplot as plt


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

def plot_scores(df_scores, rolling=100):
    df_scores_rolling = df_scores.rolling(rolling).mean()
    df_scores_rolling.plot(kind='line')

    # plt.show(block=False)
    plt.draw()
    plt.pause(0.001)