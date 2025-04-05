import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

from homework.datasets.road_dataset import RoadDataset
from homework.models import load_model, save_model


def train(model_name, num_epoch, lr, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {model_name} on {device}")

    # Load data
    dataset = RoadDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_model(model_name)
    model.to(device)

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # Training loop
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            track_left = batch.get("track_left", None)
            track_right = batch.get("track_right", None)
            image = batch.get("image", None)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            # Select model input
            if model_name == "mlp_planner" or model_name == "transformer_planner":
                inputs = {
                    "track_left": track_left.to(device),
                    "track_right": track_right.to(device),
                }
            elif model_name == "cnn_planner":
                inputs = {
                    "image": image.to(device),
                }
            else:
                raise ValueError(f"Unknown model: {model_name}")

            pred = model(**inputs)

            loss = loss_fn(pred[mask], waypoints[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epoch}: Loss = {avg_loss:.4f}")

    # Save model
    path = save_model(model)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="mlp_planner / transformer_planner / cnn_planner")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train(
        model_name=args.model_name,
        num_epoch=args.num_epoch,
        lr=args.lr,
    )
