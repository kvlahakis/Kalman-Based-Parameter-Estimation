import torch
import os
import argparse
from pathlib import Path
from enkf_ppe.Dynamics.Lorentz63 import forward


def generate_dataset(num_steps, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, initial_state=None):
    """
    Generates or extends a Lorenz '63 dataset.
    
    Args:
        num_steps (int): Number of steps to generate.
        dt (float): Time step size.
        sigma (float): Lorenz parameter sigma.
        rho (float): Lorenz parameter rho.
        beta (float): Lorenz parameter beta.
        initial_state (torch.Tensor, optional): Starting state if the file doesn't exist.
    """
    metadata = {
        'num_steps': num_steps,
        'parameters': {
            'sigma': sigma,
            'rho': rho,
            'beta': beta,
        },
        'dt': dt,
        'initial_state': initial_state.tolist()
    }
    
    # Generate filename from metadata
    filename = f"sigma{sigma:.4f}_rho{rho:.4f}_beta{beta:.4f}_dt{dt:.4f}.pt"
    file_path = Path(__file__).parent / filename

    # 1. Load existing data if it exists
    if file_path.exists():
        payload = torch.load(file_path, weights_only=True)

        
        if isinstance(payload, dict) and 'data' in payload:
            existing_data = payload['data']
            existing_metadata = payload.get('metadata', {})
            
            # Check if metadata matches
            for key in metadata:
                if key in existing_metadata and existing_metadata[key] != metadata[key]:
                    print(f"Warning: Metadata mismatch for {key}. Existing: {existing_metadata[key]}, New: {metadata[key]}")
            
            start_state = existing_data[-1]
            print(f"Continuing from existing data at {file_path}. Current size: {existing_data.shape[0]}")
        else:
            print(f"Warning: Existing file at {file_path} has unexpected format. Starting fresh.")
            existing_data = torch.empty((0, 3))
            start_state = initial_state if initial_state is not None else torch.tensor([1.0, 1.0, 1.0])
        
    else:
        # 2. Use initial state or a default if starting fresh
        start_state = initial_state if initial_state is not None else torch.tensor([1.0, 1.0, 1.0])
        existing_data = torch.empty((0, 3))
        print(f"Starting new dataset at {file_path}. Initial state: {start_state.tolist()}")
    
    remaining_steps = num_steps - existing_data.shape[0]
    if remaining_steps <= 0:
        print(f"Dataset already has {existing_data.shape[0]} steps. No new steps will be generated.")
        return

    # 3. Generate new points
    new_points = []
    current_state = start_state
    
    theta = torch.tensor([sigma, rho, beta], dtype=torch.float32)

    for _ in range(remaining_steps):
        current_state = forward(current_state, theta, dt=dt)
        new_points.append(current_state.clone())
    
    # 4. Concatenate and save
    if new_points:
        new_tensor = torch.stack(new_points)
        final_data = torch.cat([existing_data, new_tensor], dim=0)
        
        save_payload = {
            'data': final_data,
            'metadata': metadata
        }
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(save_payload, file_path)
        print(f"Dataset updated. Generated {remaining_steps} new steps. Total size: {final_data.shape[0]} points.")
        print(f"Saved to: {file_path}")
    else:
        print("No steps were generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or extend a Lorenz '63 dataset.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of integration steps to generate.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument("--sigma", type=float, default=10.0, help="Lorenz parameter sigma.")
    parser.add_argument("--rho", type=float, default=28.0, help="Lorenz parameter rho.")
    parser.add_argument("--beta", type=float, default=8/3, help="Lorenz parameter beta.")
    parser.add_argument("--x0", type=float, nargs=3, default=[12.0, 20.0, 5.0], help="Initial state [x, y, z] if starting fresh.")

    args = parser.parse_args()
    
    initial_state = torch.tensor(args.x0, dtype=torch.float32)
    
    generate_dataset(
        num_steps=args.steps,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        initial_state=initial_state
    )
