def epoch_plot(epochs, type: str, combine_strategy: str):
    epochs[type].plot_image(
    combine=combine_strategy,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
    )
    
    