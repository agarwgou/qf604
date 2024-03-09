import matplotlib.pyplot as plt
import numpy as np

# Plot 
# Generate an array of time intervals

def plot_eda(df, study = 'replicate'):
    if study not in ['replicate', 'extension']:
        raise ValueError('Study has to be either "replicate" or "extension"!')
    
    #Descriptive plot, replicating paper's
    if study == 'replicate':
        figsize = (16,12)
        grid_size = (2, 2)
    else:
        figsize = (16,16)
        grid_size = (3, 2)

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(grid_size, (1, 0))
    ax3 = plt.subplot2grid(grid_size, (1, 1))

    df['daily_return'].plot(ax = ax1)
    ax1.set_title('Daily Return of NYMEX Natural Gas Futures')
    
    df['Monthly_Disaster_Freq'].resample('M').mean().plot(ax=ax2)
    ax2.set_title('Monthly Frequency of Natural Diasters')
    
    df['cpu_index'].resample('M').mean().plot(ax=ax3)
    ax3.set_title('U.S. Climate Policy Uncertainy Index')

    if study == 'extension':
        ax4 = plt.subplot2grid(grid_size, (2, 0))
        ax5 = plt.subplot2grid(grid_size, (2, 1))

        df[['North', 'South']].resample('M').mean().plot(ax=ax4)
        ax4.set_title('Average Monthly Temperature in North and South US')
        ax4.legend()

        df['Storage'].resample('M').mean().plot(ax=ax5)
        ax5.set_title('U.S. Natural Gas Storage')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def plot_results(x, ht, tau, title = None):

    plt.plot(x, np.sqrt(ht), 'g--', label = r'$\sqrt{h_t}$')
    plt.plot(x, np.sqrt(tau),'b-', label = r'$\sqrt{\tau_t}$')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_forecast(x, realized_vol, forecast_vol, title = None):
    plt.plot(x, realized_vol, 'g--', label = 'intraday vol')
    plt.plot(x, forecast_vol,'b-', label = 'forecasted vol')
    if title:
        plt.title(title)
        
    plt.legend()
    plt.show()