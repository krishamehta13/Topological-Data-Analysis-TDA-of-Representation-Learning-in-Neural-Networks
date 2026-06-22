import torch
import torch.nn as nn

class ActivationHook:
    """Helper class to register a forward hook and capture the activation/embeddings of a layer."""
    def __init__(self):
        self.activation = None

    def __call__(self, module, input, output):
        # Store a copy of output activations detached from the computation graph
        self.activation = output.detach().cpu().numpy()

class TopologicalMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[32, 16, 8], output_dim=2, activation='relu'):
        super(TopologicalMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.hooks = {}
        
        # Select activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh
        elif activation.lower() == 'sigmoid':
            act_fn = nn.Sigmoid
        else:
            act_fn = nn.ReLU
            
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.layers.append(act_fn())
            current_dim = h_dim
            
        # Final classification layer
        self.layers.append(nn.Linear(current_dim, output_dim))
        
        # Register hooks for each linear layer's output (before activation or after activation)
        # Let's register after the activation function for each hidden layer, and also before.
        # Capturing after activation shows the representation that is actually fed to the next layer.
        self._register_hooks()

    def _register_hooks(self):
        hook_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # We name the layer based on its index
                name = f"Layer_{hook_idx}_Linear"
                hook = ActivationHook()
                layer.register_forward_hook(hook)
                self.hooks[name] = hook
                
                # Also capture activation if the next element is an activation function
                if i + 1 < len(self.layers) and not isinstance(self.layers[i+1], nn.Linear):
                    act_name = f"Layer_{hook_idx}_{type(self.layers[i+1]).__name__}"
                    act_hook = ActivationHook()
                    self.layers[i+1].register_forward_hook(act_hook)
                    self.hooks[act_name] = act_hook
                hook_idx += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_embeddings(self):
        """Returns a dictionary containing the latest captured activations from all registered layers."""
        return {name: hook.activation.copy() for name, hook in self.hooks.items() if hook.activation is not None}
