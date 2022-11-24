import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
                num_classes=10,
                input_size=28*28,
                hidden_width=512,
                hidden_layers=1,
                dropout_p=0.5)->None:
        """
        || param num_classes: output size
        || param input_size: input size
        || param hidden_width: number of neurons in hidden layer
        || param hidden_layers: number of hidden layers
        || param dropout_p: dropout probability 0 to disable
        """
        super.__init__()

        assert(hidden_layers>0, 'Must have at least one hidden layer!')

        #Minimum of 1 hidden layer        
        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_width),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            )
        ) 
        
        #Add any extra layers programmatically
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_width, hidden_width),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=dropout_p),
                    )
                ),
            )
        
        #output layer
        layers.add_module(
            'output_layer',
            nn.Linear(hidden_width,num_classes)
        )
    
        self.model= nn.Sequential(*layers)
        
        self._input_size = input_size

    def forward(self, x) -> torch.Tensor:
        x = x.view(x.size(0), self._input_size)
        x = self.model(x)
        return x