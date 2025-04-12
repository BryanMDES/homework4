from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints


        
        input_dim = n_track * 2 * 2 # How many points we have on one side of the track
        output_dim = n_waypoints * 2 # Brain guessing where car will go

        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(),nn.Dropout(0.1), nn.Linear(256,128), nn.ReLU(),nn.Dropout(0.1),nn.Linear(128, output_dim),) #Mapping 128 to 6 outputs
    

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        def normalize(t):
          mean = t.mean(dim=1, keepdim=True)
          std = t.std(dim=1, keepdim=True) + 1e-6
          return (t - mean) / std

        # Combining both sides of the road
        track_left = normalize(track_left)
        track_right = normalize(track_right)
        x = torch.cat([track_left, track_right], dim=1)

        x = x.view(x.size(0), -1) # Model receiving one long row of numbers
        out = self.net(x) # Pass the flattened road into the MLD to guess the next way points

        return out.view(-1, self.n_waypoints, 2)



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track #Stores how many points we have per side of the track
        self.n_waypoints = n_waypoints #Controls the number of queries the model will use
        self.d_model = d_model # How wide the model's brain is for thiking
        self.input_proj = nn.Linear(2, d_model) # Take each x,y point and projects it into a d_model-dimensional vector
        self.query_embed = nn.Embedding(n_waypoints, d_model) # Creating learnable enbeddings for each waypoint
        decoder_layer = nn.TransformerDecoderLayer(d_model = d_model, nhead=n_heads, batch_first=True) # Defines one layer of the Transfrmer decoder
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers) # Stacks multiple decoder layers together to build a deep network
        self.output_proj = nn.Linear(d_model, 2) # This is what gets returned as the predicted waypoint

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0] 
        track = torch.cat([track_left, track_right], dim =1) # Combining the left annd right boundaries into one tensor
        track_feat = self.input_proj(track) # Projecting the points into a d-mmodel dimensional space
        query_indices = torch.arange(self.n_waypoints, device=track.device).unsqueeze(0).expand(B, -1) # Creating waypooint IDs
        queries = self.query_embed(query_indices) # Turning the waypoint ids into learned vectors
        attended = self.decoder(queries, track_feat) # Matches each query with the track features
        out = self.output_proj(attended) #X,y coordinates for the predictedwaypoints
        return out
        



class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Storing the image mean and standard deviation
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        

        # Features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.head = nn.Sequential(
            nn.Flatten(), # Shape
            nn.Linear(128 * 12 * 16, 128), # Hidden Layer
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )
        

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # None, ;, None, None] reshapes the buffers to match the image shape
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.conv_layers(x) # Extracting the features.
        x = self.head(x) # Flatten and process features through the fully connected layers to 
        return x.view(-1, self.n_waypoints, 2) # Resahpe the outptu

        #raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
