import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plotter(df_week: pd.DataFrame, height: float, width: float):
    row_heights = [0.6, 0.1, 0.1, 0.1, 0.1] 
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=row_heights)
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df_week.index,
                    open=df_week['open'],
                    high=df_week['high'],
                    low=df_week['low'],
                    close=df_week['close'], 
                    increasing_line_color='green',
                    decreasing_line_color='red'),
                    row=1, col=1)
    # Add horizontal line for the current closing price 
    fig.add_hline(y=df_week['close'].iloc[-1], line_dash="dashdot", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
    # Let's remove the gaps between sessions.
    # Update layout to remove x-axis gaps
    fig.update_xaxes(type='category', rangeslider_visible=False)
    

    # Add MAs on the candlestick chart
    # Add 5-day Moving Average Trace
    fig.add_trace(go.Scatter(x=df_week.index, 
                             y=df_week['MA50'], 
                             opacity=0.7, 
                             line=dict(color='blue', width=1), 
                             name='MA 50'),
                             row=1, col=1)
    # Add 20-day Moving Average Trace
    fig.add_trace(go.Scatter(x=df_week.index, 
                             y=df_week['MA200'], 
                             opacity=0.7, 
                             line=dict(color='orange', width=1), 
                             name='MA 200'),
                             row=1, col=1)
    
    
    # Plot the volume
    # Plot volume trace on 2nd row  
    colors = ['red' if row['open'] - row['close'] >= 0 
              else 'green' for index, row in df_week.iterrows()]
    fig.add_trace(go.Bar(x=df_week.index,
                         y=df_week['volume'],
                         marker_color=colors
                        ), row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=2, col=1)
    
    
    # Plot MACD trace on 3rd row
    colors = ['green' if val >= 0 
              else 'red' for val in df_week.macd_diff]
    fig.add_trace(go.Bar(x=df_week.index,
                         y=df_week.macd_diff,
                         marker_color=colors
                        ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_week.index,
                             y=df_week.macd,
                             #opacity=0.7,
                             line=dict(color='#03A8FF', width=0.5)
                            ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_week.index,
                             y=df_week.macd_signal,
                             #opacity=0.7,
                             line=dict(color='#FF9300', width=0.5)
                            ), row=3, col=1)
    fig.update_yaxes(showgrid=True, dtick=0.5, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=3, col=1)
    fig.add_hline(y=1, line_dash="dash", row=3, col=1)
    fig.add_hline(y=-1, line_dash="dash", row=3, col=1)
    # Adjust the y-axis range
    fig.update_yaxes(range=[-2.0, 2.0], row=5, col=1)  # Set y-axis range from 15 to 25
    
    
    # Plot stochastics trace on 4th row
    fig.add_trace(go.Scatter(x=df_week.index,
                             y=df_week.stoch,
                             line=dict(color='#078DFD', width=0.5)
                            ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_week.index,
                             y=df_week.stoch_signal,
                             line=dict(color='#FA1D1D', width=0.5)
                            ), row=4, col=1)
    fig.update_yaxes(showgrid=True, dtick=20, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", row=4, col=1)
    # Adjust the y-axis range
    fig.update_yaxes(range=[0.0, 100.0], row=5, col=1)  # Set y-axis range from 15 to 25
    
    
    # Plot the time of the session 
    #df_week['time'] = df_week.index.apply(lambda x: x.time())
    #df_week['time'] = df_week.index.to_series().apply(lambda x: x.time())
    
    fig.add_trace(go.Bar(x=df_week.index,
                         y=df_week['time'],
                         marker_color='#FD8607'
                         ),
                         row=5, col=1)
    # Add horizontal lines correspond to beginning and end of the session
    fig.add_hline(y=0.0, line_dash="dash", row=5, col=1)
    fig.add_hline(y=10.0, line_dash="dash", row=5, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', row=5, col=1)
    # Adjust the y-axis range
    fig.update_yaxes(range=[-2, 12], row=5, col=1)  # Set y-axis range from 15 to 25
    
    
    # Remove tick values on x and y axises.
    for row in range(1, 5+1):
        fig.update_xaxes(showticklabels=False, row=row, col=1)
        fig.update_yaxes(showticklabels=False, row=row, col=1)

    # Update layout to set background color to white
    fig.update_layout(
        paper_bgcolor='white',  # Set the background color
        plot_bgcolor='white',    # Set the plot area background color
        height=height, width=width, 
        showlegend=False, xaxis_rangeslider_visible=False
    )

    return fig