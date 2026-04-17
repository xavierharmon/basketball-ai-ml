# Neural Networks Learning Project

A comprehensive learning project combining neural network fundamentals with NCAA basketball data analysis.

## Project Structure

```
ai-ml/
├── data/
│   ├── ncaa_basketball/
│   │   ├── raw/              # Downloaded raw data
│   │   └── processed/        # Cleaned and processed data
│   ├── kenpom/               # KenPom ratings
│   └── nil/                  # NIL data
├── notebooks/
│   ├── 01_neural_networks_fundamentals/
│   │   ├── 01_intro_to_neural_networks.ipynb
│   │   ├── 02_forward_propagation.ipynb
│   │   ├── 03_backpropagation.ipynb
│   │   ├── 04_activation_functions.ipynb
│   │   └── 05_optimization_and_training.ipynb
│   ├── 02_data_exploration/
│   │   └── 01_ncaa_data_overview.ipynb
│   └── 03_architectures/
│       ├── 01_feedforward_network.ipynb
│       ├── 02_cnn_basics.ipynb
│       └── 03_rnn_lstm_basics.ipynb
├── src/
│   ├── nn_core/
│   │   ├── activations/      # Activation functions
│   │   ├── layers/           # Layer implementations
│   │   └── __init__.py
│   └── utils/                # Data processing utilities
├── docs/                     # Documentation
├── results/                  # Outputs, visualizations
└── README.md                 # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter scikit-learn
pip install torch torchvision  # For PyTorch (optional, for advanced examples)
```

### 2. Download Data

Visit [Kaggle Datasets](https://www.kaggle.com/datasets?search=NCAA+basketball) and download:

- NCAA Regular Season Basketball Games (2011-current)
- KenPom Ratings 2025
- March Madness Historical Dataset
- College Basketball Dataset

Place downloaded files in:
- `data/ncaa_basketball/raw/` - Game and team data
- `data/kenpom/` - KenPom ratings
- `data/nil/` - NIL data

### 3. Start Learning

#### Phase 1: Neural Networks Fundamentals (Weeks 1-2)
Start with these notebooks in order:
1. `01_intro_to_neural_networks.ipynb` - Basic concepts
2. `02_forward_propagation.ipynb` - How predictions are made
3. `03_backpropagation.ipynb` - How networks learn
4. `04_activation_functions.ipynb` - Non-linearity in networks
5. `05_optimization_and_training.ipynb` - Training algorithms

#### Phase 2: Data Exploration (Week 2-3)
- `02_ncaa_data_overview.ipynb` - Understanding basketball data

#### Phase 3: Practical Applications (Week 3+)
- `01_feedforward_network.ipynb` - Simple predictions with basketball data
- `02_cnn_basics.ipynb` - Image-based features (optional)
- `03_rnn_lstm_basics.ipynb` - Time-series predictions

## Key Concepts Covered

### Neural Network Fundamentals
- **Neurons and Layers**: Building blocks of neural networks
- **Forward Propagation**: How data flows through the network
- **Loss Functions**: Measuring prediction error
- **Backpropagation**: How networks learn from errors
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax

### Training & Optimization
- **Gradient Descent**: Basic optimization algorithm
- **Stochastic Gradient Descent (SGD)**: Faster training
- **Momentum**: Accelerating convergence
- **Adam Optimizer**: Adaptive learning rates

### Architectures
- **Feedforward Networks**: Simple neural networks
- **Convolutional Neural Networks (CNNs)**: For spatial data
- **Recurrent Neural Networks (RNNs)**: For sequences
- **LSTM**: Long-term dependencies

### Data Analysis
- NCAA basketball game data
- Team statistics and rankings
- KenPom efficiency ratings
- Predictive modeling

## Code Examples

### Using Custom Activation Functions
```python
from src.nn_core.activations import sigmoid, relu, softmax
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = relu(x)  # [1.0, 2.0, 3.0]
```

### Using Dense Layer
```python
from src.nn_core.layers import DenseLayer

layer = DenseLayer(input_size=10, output_size=5, activation='relu')
output = layer.forward(input_data)
```

### Loading Data
```python
from src.utils import load_ncaa_data

data = load_ncaa_data('data/ncaa_basketball/raw')
```

## Learning Path

**Week 1: Fundamentals**
- Understand what neural networks are
- Learn forward and backward propagation
- Understand activation functions and their derivatives

**Week 2: Training**
- Learn about loss functions
- Understand gradient descent variants
- Implement simple networks from scratch

**Week 3: Applications**
- Use PyTorch for practical implementations
- Work with NCAA basketball data
- Build predictive models

**Week 4+: Advanced Topics**
- Explore CNNs and RNNs
- Fine-tune on basketball data
- Experiment with architectures

## Resources

### Online Learning
- [3Blue1Brown - Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000D_yBMXo36wV)
- [Coursera - Machine Learning](https://www.coursera.org/specializations/machine-learning-introduction)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai)

### Papers & Articles
- Backpropagation: [Rumelhart et al., 1986](https://www.nature.com/articles/323533a0)
- Adam Optimizer: [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- CNNs: [LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

## Tips for Learning

1. **Run the code**: Don't just read, execute and modify the notebooks
2. **Experiment**: Change parameters and see how they affect results
3. **Visualize**: Use the plotting code to understand what's happening
4. **Build intuition**: Why does this work? Why does that fail?
5. **Connect concepts**: Relate concepts to basketball (or your domain)

## Next Steps

- [ ] Download NCAA basketball datasets
- [ ] Complete fundamentals notebooks
- [ ] Implement your own neural network
- [ ] Build a prediction model with basketball data
- [ ] Experiment with different architectures
- [ ] Optimize hyperparameters

## Questions?

When something is unclear:
1. Review the relevant notebook
2. Check the code comments
3. Look at online resources
4. Experiment and trace through the code

## License

This project is created for educational purposes.

---

**Happy Learning!** 🧠🏀
