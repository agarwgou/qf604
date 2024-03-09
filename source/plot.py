import matplotlib.pyplot as plt
import numpy as np

# Plot 
# Generate an array of time intervals

def plot_eda(df):
    #Descriptive plot, replicating paper's
    fig = plt.figure(figsize=(16, 12))

    # Define grid size and positions
    grid_size = (2, 2)
    ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(grid_size, (1, 0))
    ax3 = plt.subplot2grid(grid_size, (1, 1))

    df['daily_return'].plot(ax = ax1)
    ax1.set_title('Daily Return of NYMEX Natural Gas Futures')
    
    df['Monthly_Disaster_Freq'].resample('M').mean().plot(ax=ax2)
    ax2.set_title('Monthly Frequency of Natural Diasters')
    
    df['cpu_index'].resample('M').mean().plot(ax=ax3)
    ax3.set_title('U.S. Climate Policy Uncertainy Index')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def plot_results(x, ht, tau, title = None):

    plt.plot(x, np.sqrt(ht), 'g--', label = r'$\sqrt{h_t}$')
    plt.plot(x, np.sqrt(tau),'b-', label = r'$\sqrt{\tau_t}$')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()