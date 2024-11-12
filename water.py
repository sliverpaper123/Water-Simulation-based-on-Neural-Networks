# Neural Water Simulation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
import torch
import torch.nn as nn
import numpy as np

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define grid size and time steps
grid_size = 300
time_steps = 900
dt = 0.005

# Neural Network for Wave Propagation
class WaveNet(nn.Module): 
    def __init__(self, grid_size):
        super(WaveNet, self).__init__()
        
        # Convolutional layers for spatial features
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
        # Initialize weights with small values for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, current, previous):
        # Stack current and previous states
        x = torch.cat([current, previous, current - previous], dim=1)
        # Predict wave propagation
        delta = self.spatial_features(x)
        # Apply physical constraints
        new_state = 2 * current - previous + 0.1 * delta
        return new_state * 0.9905  # Apply damping

# Initialize neural network
model = WaveNet(grid_size).to(device)

# Initialize water grids
water_grid = torch.zeros((1, 1, grid_size, grid_size), device=device)
prev_grid = torch.zeros((1, 1, grid_size, grid_size), device=device)

# Create a smooth Gaussian drop
def create_gaussian_drop(size, center, sigma=4.0, amplitude=1.0):
    x = torch.arange(0, size, device=device)
    y = torch.arange(0, size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    dist_sq = (X - center[0])**2 + (Y - center[1])**2
    return amplitude * torch.exp(-dist_sq / (2 * sigma**2))

# Apply initial drop
center = (grid_size // 2, grid_size // 2)
drop = create_gaussian_drop(grid_size, center)
water_grid[0, 0] = drop

# Training data generation for physical constraints
def generate_training_data(num_samples=1000):
    training_data = []
    labels = []
    
    for _ in range(num_samples):
        # Random initial conditions
        x = torch.randn(1, 1, grid_size, grid_size, device=device) * 0.1
        prev = torch.randn(1, 1, grid_size, grid_size, device=device) * 0.1
        
        # Classical wave equation solution
        laplacian = nn.functional.conv2d(x, 
            torch.tensor([[[[0, 0.1, 0],
                          [0.1, -0.4, 0.1],
                          [0, 0.1, 0]]]], device=device),
            padding=1)
        next_state = 2 * x - prev + laplacian
        
        training_data.append((x, prev))
        labels.append(next_state)
    
    return training_data, labels

# Train the model
def train_model(model, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training neural network...")
    training_data, labels = generate_training_data()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for (current, prev), target in zip(training_data, labels):
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(current, prev)
            
            # Calculate loss
            loss = criterion(predicted, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(training_data):.6f}")

# Train the model
train_model(model)

# Visualization setup
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')
plt.tight_layout()

# Generate meshgrid for plotting
X, Y = np.meshgrid(np.linspace(0, grid_size, grid_size),
                   np.linspace(0, grid_size, grid_size))

# Create LightSource for shading
ls = LightSource(azdeg=315, altdeg=45)

# Initial plot
data_initial = water_grid.cpu().numpy().squeeze()
rgb_initial = ls.shade(data_initial, cmap=cm.viridis, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(X, Y, data_initial, facecolors=rgb_initial,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)

# Set plot parameters
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_zlim(-10, 10)
ax.set_title('Neural Water Simulation - Time Step: 0', fontsize=18)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Animation update function
def update(frame):
    global water_grid, prev_grid, surf
    
    with torch.no_grad():
        # Use neural network to predict next state
        new_grid = model(water_grid, prev_grid)
        
        # Apply shoreline damping
        shoreline_x = int(grid_size * 0.9)
        mask = torch.linspace(0, grid_size, grid_size, device=device) > shoreline_x
        mask = mask.view(1, 1, 1, -1)
        new_grid = torch.where(mask, new_grid * 0.9, new_grid)
        
        # Update grids
        prev_grid, water_grid = water_grid, new_grid
    
    # Convert to CPU for plotting
    data = water_grid.cpu().numpy().squeeze()
    data[:, int(grid_size * 0.9):] = 0
    
    # Update shading and surface
    rgb = ls.shade(data, cmap=cm.viridis, vert_exag=0.1, blend_mode='soft')
    surf.remove()
    surf = ax.plot_surface(X, Y, data, facecolors=rgb,
                          rstride=1, cstride=1, linewidth=0, antialiased=False)
    
    ax.set_title(f'Neural Water Simulation - Time Step: {frame}', fontsize=18)
    return surf,

# Create and save animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=30, blit=False)
Writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save('neural_water_simulation.mp4', writer=Writer)

plt.show()  # Uncomment to display instead of saving