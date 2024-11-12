# Neural Water Simulation

An advanced water surface simulation using deep learning and PyTorch. This project combines neural networks with classical wave equations to create realistic water wave animations.

![Water Simulation Example](demo.gif)

## 🌊 Overview

This project implements a neural network-based water simulation that:
- Uses deep learning to predict wave propagation
- Combines physical wave equations with neural predictions
- Generates realistic water surface animations
- Supports GPU acceleration through CUDA
- Includes realistic effects like shoreline damping and wave interactions

## 🚀 Features

- **Neural Network Wave Prediction**: Uses a custom WaveNet architecture to learn and predict wave patterns
- **Physics-Based Training**: Combines machine learning with classical wave equations
- **Real-time 3D Visualization**: Creates smooth, animated visualizations of the water surface
- **GPU Acceleration**: CUDA support for faster computation
- **Customizable Parameters**: Easily adjust simulation size, time steps, and wave properties
- **Realistic Effects**: Includes shoreline damping, wave reflection, and natural wave decay

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)
- matplotlib
- numpy

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-water-simulation.git
cd neural-water-simulation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the simulation:
```bash
python water.py
```

2. Customize parameters in the script:
```python
# Adjust simulation parameters
grid_size = 300  # Size of the simulation grid
time_steps = 900  # Number of simulation steps
dt = 0.005  # Time step size
```

3. The simulation will generate an MP4 file showing the water surface animation.

## 🔧 Configuration

Key parameters that can be modified:

```python
# Grid and simulation parameters
grid_size = 300  # Size of water surface
time_steps = 900  # Length of simulation
dt = 0.005  # Time step size

# Initial drop parameters
sigma = 4.0  # Drop spread
amplitude = 1.0  # Drop height

# Neural network parameters
learning_rate = 0.001
num_epochs = 50
num_training_samples = 1000
```

## 🧪 How It Works

1. **Neural Network Architecture**
   - Uses a WaveNet model with convolutional layers
   - Processes spatial features of the water surface
   - Predicts wave propagation patterns

2. **Wave Physics Integration**
   - Combines neural predictions with classical wave equations
   - Applies physical constraints for realistic behavior
   - Includes damping and boundary effects

3. **Visualization**
   - Creates 3D surface plots using matplotlib
   - Updates in real-time during simulation
   - Includes dynamic lighting and color effects

## 📊 Technical Details

### WaveNet Architecture
```python
class WaveNet(nn.Module):
    def __init__(self, grid_size):
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # ... more layers for wave processing
        )
```

### Physics Integration
```python
def forward(self, current, previous):
    # Combines neural prediction with wave equation
    new_state = 2 * current - previous + 0.1 * delta
    return new_state * 0.9905  # Apply damping
```

## 🎯 Future Improvements

- [ ] Add support for multiple simultaneous drops
- [ ] Implement interactive real-time visualization
- [ ] Add wind effects and wave interference patterns
- [ ] Optimize performance for larger grid sizes
- [ ] Add support for custom boundary conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
