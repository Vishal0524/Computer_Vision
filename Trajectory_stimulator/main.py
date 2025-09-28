"""
3D Bee Trajectory Simulation using Lorenz Equations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


class BeeTrajectorySimulator:
    """Simulate and visualize 3D bee movement using Lorenz equations."""
    
    def __init__(self, a=10, b=28, c=2.667, initial_state=(0, 1, 1.05)):
        self.a = a
        self.b = b
        self.c = c
        self.initial_state = np.array(initial_state)
        self.trajectory = None
        self.time_points = None
    
    def lorenz_system(self, t, state):
        """Lorenz differential equations."""
        x, y, z = state
        return [
            self.a * (y - x),
            self.b * x - y - x * z,
            x * y - self.c * z
        ]
    
    def simulate(self, t_span=(0, 30), num_points=10000):
        """Simulate bee trajectory."""
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(
            self.lorenz_system,
            t_span,
            self.initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )
        
        self.time_points = solution.t
        self.trajectory = solution.y.T
        return self.trajectory, self.time_points
    
    def plot_3d_trajectory(self, figsize=(12, 9)):
        """Create 3D visualization of bee path."""
        if self.trajectory is None:
            raise ValueError("Must simulate trajectory first!")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        

        ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], 
                self.trajectory[:, 2], 'steelblue', linewidth=0.6, 
                alpha=0.7, label='Bee Flight Path')
        
        # Start position
        ax.scatter(*self.initial_state, color='green', s=100,
                  label=f'Start: {tuple(self.initial_state)}')
        
        # End position
        end_pos = self.trajectory[-1]
        ax.scatter(*end_pos, color='red', s=100,
                  label=f'End: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})')
        
        # recent_steps = min(2000, len(self.trajectory) // 4)
        # ax.plot(self.trajectory[-recent_steps:, 0], 
        #         self.trajectory[-recent_steps:, 1], 
        #         self.trajectory[-recent_steps:, 2], 
        #         'orange', linewidth=1.5, alpha=0.9, label='Recent Path')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Bee Trajectory - Lorenz Attractor\n'
                    f'Parameters: a={self.a}, b={self.b}, c={self.c}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_time_series(self, figsize=(12, 8)):
        """Plot position components over time."""
        if self.trajectory is None:
            raise ValueError("Must simulate trajectory first!")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        components = ['X', 'Y', 'Z']
        colors = ['blue', 'green', 'red']
        
        # Individual component plots
        for i, (comp, color) in enumerate(zip(components, colors)):
            if i < 3:
                row, col = divmod(i, 2)
                ax = axes[row, col]
                ax.plot(self.time_points, self.trajectory[:, i],
                       color=color, linewidth=0.8, alpha=0.8)
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{comp} Position')
                ax.set_title(f'{comp} Component')
                ax.grid(True, alpha=0.3)
        
        # Combined plot
        ax = axes[1, 1]
        for i, (comp, color) in enumerate(zip(components, colors)):
            ax.plot(self.time_points, self.trajectory[:, i],
                   color=color, linewidth=0.8, alpha=0.7, label=f'{comp}(t)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title('All Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def analyze_trajectory(self):
        """Statistical analysis of trajectory."""
        if self.trajectory is None:
            raise ValueError("Must simulate trajectory first!")
        
        print("=== TRAJECTORY ANALYSIS ===")
        print(f"Parameters: a={self.a}, b={self.b}, c={self.c}")
        print(f"Initial: {tuple(self.initial_state)}")
        print(f"Time: {self.time_points[0]:.1f} to {self.time_points[-1]:.1f}")
        print(f"Points: {len(self.trajectory)}")
        
        components = ['X', 'Y', 'Z']
        for i, comp in enumerate(components):
            data = self.trajectory[:, i]
            print(f"{comp}: mean={np.mean(data):6.2f}, "
                  f"std={np.std(data):5.2f}, "
                  f"range=[{np.min(data):6.2f}, {np.max(data):6.2f}]")


def main():
    """Main simulation function."""
    # Initialize and simulate
    simulator = BeeTrajectorySimulator()
    trajectory, time_points = simulator.simulate()
    
    # Create visualizations
    fig1, ax1 = simulator.plot_3d_trajectory()
    fig2, axes2 = simulator.plot_time_series()
    
    # Analysis
    simulator.analyze_trajectory()
    
    plt.show()
    return simulator


if __name__ == "__main__":
    bee_sim = main()