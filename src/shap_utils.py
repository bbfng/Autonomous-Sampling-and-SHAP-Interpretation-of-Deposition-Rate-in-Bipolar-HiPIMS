import shap
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import ticker

import numpy as np

import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

names = {
    'PW (us)': 'neg. PW (\u03BCs)',
    'PRR (Hz)': 'Frequency (Hz)',
    'Ipk (A)': '$J_{\mathrm{pk}}$ (A cm$^{-2}$)',
    'pos. Delay (us)': 'pos. Delay (\u03BCs)',
    'pos. PW (us)': 'pos. PW (\u03BCs)',
    'pos. Setpoint (V)': 'pos. Voltage (V)',
    'Power (W)': 'Power (W cm$^{-2}$)',
    'y1': 'Deposition Rate (\u212B s$^{-1}$)',
    'metal_type':'Al or Ti'
}

def xy_SHAP_plot(df_raw,df_SHAP,feature,feature_comp,ax,
                  color = 'magma',
                  show_ylabel = True,
                  truncate_color = False,
                  cbar_label: str = 'output',
                  title: str | None = None,
                   fontsize:int=8,
                   s:int = 40) -> None:

    if truncate_color == True:
        #remove the yellow of the colormap
        base_cmap = plt.cm.get_cmap(color)
        truncated_cmap = LinearSegmentedColormap.from_list(
        "cmap_trunc",
        base_cmap(np.linspace(0.0, 0.8, 256))
        )
    #Plot against color
    sc = ax.scatter(df_raw[feature], df_raw[feature_comp], c=df_SHAP[feature], cmap=truncated_cmap, s=s, edgecolor = 'white',linewidth = 0.5)
    #add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('SHAP Value of ' + cbar_label + ' (\u212B s$^{-1}$)', rotation=270, labelpad=15, fontsize = fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    
    ax.set_xlabel(names[feature],fontsize = fontsize)
    if show_ylabel == True:
        ax.set_ylabel(names[feature_comp],fontsize = fontsize)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))  # 5 minor ticks between majors
    ax.tick_params(direction='in',top=True,right=True)
    ax.tick_params(which='minor',direction='in',top=True,right=True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-1)

    if title is not None:
        plt.title(title, fontsize=fontsize+2)

def plot_dependencies(
        explanation,
        feature_name,
    ):
    other_feature_names = [
        name for name in explanation.feature_names if name != feature_name
    ]
    for feature in other_feature_names:
        shap.plots.scatter(
            explanation[:, feature_name],
            color=explanation[:, feature],
        )
        plt.show()


def dependence_plot_single(
        df_raw,
        df_SHAP,
        feature,
        feature_comp,
        ax,
        ylim:tuple[float,float] | None = None,
        ylabel:str=None,
        show_y_labels: bool=True,
        fontsize:int=8,
        color:str='magma',
        show_colorbar:bool=False,
        title:str | None = None,
        s:int = 40,
        box_aspect:bool=False
    ) -> plt.Axes:
    """
    Plot a SHAP dependence plot for a single feature with customizations.

    Args:
        df_raw: DataFrame containing raw feature values.
        df_SHAP: DataFrame containing SHAP values.
        feature: The feature to plot.
        feature_comp: The feature to color by.
        ax: Matplotlib Axes to plot on.
        ylim: Tuple specifying y-axis limits.
        ylabel: Label for the y-axis.
        show_y_labels: Whether to show y-axis labels.
        fontsize: Font size for labels and ticks.

    Returns:
        Matplotlib Axes with the dependence plot.
    """


    #Plot against color
    sc = ax.scatter(df_raw[feature], df_SHAP[feature], c=df_raw[feature_comp], cmap=color, s=s, edgecolor = 'white',linewidth = 0.5)

    ax.set_ylabel(ylabel,fontsize = fontsize)
    ax.set_xlabel(names[feature], fontsize = fontsize)
    if ylim == None :
        ylim = [ax.get_ylim()[0],ax.get_ylim()[1]]
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(ylim)

    if show_y_labels == False:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('SHAP Value (\u212B s$^{-1}$)',fontsize = fontsize)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))  # 5 minor ticks between majors
    ax.tick_params(direction='in',top=True,right=True)
    ax.tick_params(which='minor',direction='in',top=True,right=True)
    ax.tick_params(which='both',labelsize = fontsize - 2)

    if box_aspect:
        ax.set_box_aspect(1)
        
    if show_colorbar == True:
        cbar = plt.colorbar(sc)
        cbar.set_label(names[feature_comp], rotation=270, labelpad=15, fontsize = fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 2) 

    if title is not None:
        ax.set_title(title, fontsize=fontsize+2)
        
    #plot dotted line at mean
    x = np.linspace(-10,1.2*np.max(df_raw[feature]),1000)
    mean = [0]*1000
    sns.lineplot(x=x,y=mean,ax = ax,linestyle=':',color='black',linewidth=1.5,zorder=0)
    ax.set_xlim([df_raw[feature].min()-5,df_raw[feature].max()+5]) #adjust xlim after adding the mean line
    
    #add barchart overlay for data distribution
    #bin the data to get an idea for sampling
    num_bins = 5
    bins = np.linspace(df_raw[feature].min(),df_raw[feature].max(),num_bins + 1)
    counts, _ = np.histogram(df_raw[feature],bins=bins)
    bin_centers =  0.5 * (bins[:-1] + bins[1:])

    #normalize the bins to the data_set
    counts = counts*(ylim[1]*0.5/counts.max())
    bar_width = (bins[1]-bins[0])
    ymin = ylim[0]
    ax.bar(bin_centers,counts,width=bar_width,color='grey',alpha=0.2,bottom = ymin)

    return ax


def plot_single(
        df_raw,
        df_SHAP,
        feature,
        ax,
        ylim:tuple[float,float] | None = None,
        ylabel:str=None,
        show_y_labels: bool=True,
        fontsize:int=8,
        title:str | None = None,
        line_color:str='black',
        label:str | None = None,
        cmap = None,
        c_data = None
    ) -> plt.Axes:
    """
    Plot a SHAP dependence plot for a single feature with customizations.

    Args:
        df_raw: DataFrame containing raw feature values.
        df_SHAP: DataFrame containing SHAP values.
        feature: The feature to plot.
        feature_comp: The feature to color by.
        ax: Matplotlib Axes to plot on.
        ylim: Tuple specifying y-axis limits.
        ylabel: Label for the y-axis.
        show_y_labels: Whether to show y-axis labels.
        fontsize: Font size for labels and ticks.

    Returns:
        Matplotlib Axes with the dependence plot.
    """


    #Plot against color
    if c_data is not None and cmap is not None:
        kwargs = {'c':c_data,'cmap':cmap}
    else:
        kwargs = {'color':line_color}
    sc = ax.scatter(df_raw[feature], df_SHAP[feature], s=40, edgecolor = 'white',linewidth = 0.5, label = label, **kwargs)

    #add colorbar if data provided
    if c_data is not None and cmap is not None:
        divider = make_axes_locatable(ax) #create divider for existing axis
        cax = divider.append_axes("right", size="2%", pad=0.1) #append new axis for colorbar

        cbar = plt.colorbar(sc, cax=cax, orientation = 'vertical', aspect  =50)

        vmin, vmax = sc.get_clim() #set ticks to only min and max
        cbar.set_ticks([vmin, vmax])




    ax.set_ylabel(ylabel,fontsize = fontsize)
    ax.set_xlabel(names[feature], fontsize = fontsize)
    if ylim == None :
        ylim = [ax.get_ylim()[0],ax.get_ylim()[1]]
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(ylim)

    if show_y_labels == False:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('SHAP Value (\u212B s$^{-1}$)',fontsize = fontsize)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))  # 5 minor ticks between majors
    ax.tick_params(direction='in',top=True,right=True)
    ax.tick_params(which='minor',direction='in',top=True,right=True)
    ax.tick_params(which='both',labelsize = fontsize - 2)

    if title is not None:
        ax.set_title(title, fontsize=fontsize+2)
        
    #plot dotted line at mean
    x = np.linspace(-10,1.2*np.max(df_raw[feature]),1000)
    mean = [0]*1000
    sns.lineplot(x=x,y=mean,ax = ax,linestyle=':',color='black',linewidth=1.5,zorder=0)
    ax.set_xlim([df_raw[feature].min()-5,df_raw[feature].max()+5]) #adjust xlim after adding the mean line
    
    #add barchart overlay for data distribution
    #bin the data to get an idea for sampling
    num_bins = 5
    bins = np.linspace(df_raw[feature].min(),df_raw[feature].max(),num_bins + 1)
    counts, _ = np.histogram(df_raw[feature],bins=bins)
    bin_centers =  0.5 * (bins[:-1] + bins[1:])

    #normalize the bins to the data_set
    counts = counts*(ylim[1]*0.5/counts.max())
    bar_width = (bins[1]-bins[0])
    ymin = ylim[0]
    ax.bar(bin_centers,counts,width=bar_width,color='grey',alpha=0.2,bottom = ymin)

    #adjust label
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize-2, handletextpad = 0.05) #changes dist. of circle and label
    return ax

def beeswarm_plot(ax,explanation_display):

    heatmap_cbar_kwargs = {
    'orientation': 'vertical',
    'location': 'right',
    'pad': 0.02,
    }
    
    # --- SHAP beeswarm setup ---
    orig_names = getattr(explanation_display, "feature_names", None)
    display_names = [names.get(n, n) for n in orig_names]
    explanation_display.feature_names = display_names

    ax = shap.plots.beeswarm(
        explanation_display,
        plot_size=None,
        show=False,
        color='plasma',
        alpha=0.5,
    )

    # --- Remove SHAP’s default colorbar (last axis added by shap) ---
    print(len(fig.axes))
    if len(fig.axes) > 1:
        fig.delaxes(fig.axes[1])

    # --- Add custom colorbar consistent with Seaborn for SHAP ---
    norm = mpl.colors.Normalize(
        vmin=explanation_display.values.min(),
        vmax=explanation_display.values.max()
    )
    sm = mpl.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, use_gridspec=True, **heatmap_cbar_kwargs)
    cbar.set_label("Feature value", fontsize=11, labelpad=2, rotation=270)
    cbar.ax.tick_params(labelsize=10, width=0.8, length=3)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color='black', labelcolor='black')

    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(['Low', 'High'])

    ax.xaxis.set_major_formatter(sci_text_format)
    ax.set_xlabel(r'SHAP value / $\mathrm{\AA\ s^{-1}}$')


    # --- Force scientific notation on x-axis ---
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))  # show scientific notation outside 1e-2 .. 1e2
    ax.xaxis.set_major_formatter(formatter)
    
    plt.show()

