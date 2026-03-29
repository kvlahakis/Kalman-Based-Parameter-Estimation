import torch
import argparse
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import sys

def visualize_dataset(file_path):
    """
    Creates an interactive 3D plot of a Lorenz '63 dataset using Plotly.
    
    Args:
        file_path (str): Path to the .pt file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        return

    # 1. Load the data dictionary
    try:
        payload = torch.load(file_path, weights_only=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if not isinstance(payload, dict) or 'data' not in payload:
        print(f"Error: {file_path} does not have the expected dictionary format with a 'data' key.")
        return

    data = payload['data']
    metadata = payload.get('metadata', {})

    if data.ndim != 2 or data.shape[1] != 3:
        print(f"Error: Data must be of shape (N, 3). Found {data.shape}.")
        return

    # 2. Extract x, y, z
    x = data[:, 0].cpu().numpy()
    y = data[:, 1].cpu().numpy()
    z = data[:, 2].cpu().numpy()
    steps = np.arange(len(x))

    # 3. Create the 3D plot with Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color=steps, # Color by time step
            colorscale='Viridis',
            width=3,
            showscale=True,
            colorbar=dict(title="Time Step")
        ),
        hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<br>Step: %{text}",
        text=steps,
        name="Trajectory"
    )])

    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start',
        hovertemplate="Start Point"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='End',
        hovertemplate="End Point"
    ))

    # Update layout
    title = "Lorenz '63 Trajectory"
    if metadata:
        title += f" (sigma={metadata.get('sigma')}, rho={metadata.get('rho')}, beta={metadata.get('beta')}, dt={metadata.get('dt')})"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            # Set aspect ratio to be proportional
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    print(f"Showing interactive Plotly window for {file_path}...")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Lorenz '63 dataset in 3D using Plotly.")
    parser.add_argument("--file", type=str, default="sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt", help="Path to the .pt file.")
    
    args = parser.parse_args()
    visualize_dataset(args.file)
