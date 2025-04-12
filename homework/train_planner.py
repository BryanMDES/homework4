import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import HuberLoss
from torch.optim import Adam
from homework.datasets.road_dataset import load_data
from homework.models import load_model, save_model

"""""""""""""""""
Training MLP, Transformer, CNN to predict waypoints and learning over many epochs

"""""""""""""""""

def train(model_name, num_epoch, lr, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    transform_pipeline = ("state_only" if model_name in ["mlp_planner", "transformer_planner"] else "image_only")    
    dataloader = load_data(
      dataset_path="/content/homework4/drive_data/train",
      transform_pipeline=transform_pipeline,  # Load validation images in batches of 32, no random changes here because we are using this to test on how good th model is
      return_dataloader=True,
      batch_size=batch_size,
      shuffle=True,)


    model = load_model(model_name)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = HuberLoss(delta=1.0)

    for epoch in range(num_epoch): 
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            track_left = batch.get("track_left", None) # Getting the input data
            track_right = batch.get("track_right", None) # Getting inputs Data
            image = batch.get("image", None) # GEtting input Data
            waypoints = batch["waypoints"].to(device) 
            mask = batch["waypoints_mask"].to(device)

            
            if model_name in ["mlp_planner", "transformer_planner"]:
              if track_left is None or track_right is None:
                print("Missing track data")
                continue
              
              track_left = track_left.to(device)
              track_right = track_right.to(device)
              inputs = {
                "track_left": track_left,
                "track_right": track_right,
        }
            elif model_name == "cnn_planner":
              if image is None:
                print("Missing image data")
                continue
              inputs = {
                "image": image.to(device),
        }
            else:
              raise ValueError(f"Mispelled: {model_name}")
            
            
            pred = model(**inputs) # Getting the predicted waypint 
            
            pred_flat = pred.view(-1, 2) # Flatten predictions
            waypoints_flat = waypoints.view(-1, 2)
            mask_flat = mask.view(-1).bool() # Flatten the mask from (B, waypoints)

            # Apply the mask to select only valid waypoints
            pred_masked = pred_flat[mask_flat]
            waypoints_masked = waypoints_flat[mask_flat]

            # Compute loss onlyon the valid masked waypoints
            loss = loss_fn(pred_masked, waypoints_masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate total loss for this epoch (for monitoring)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epoch}: Loss = {avg_loss:.4f}")

    path = save_model(model)
    print(f"Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="mlp_planner / transformer_planner / cnn_planner")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)


    args = parser.parse_args()
    train(
        model_name=args.model_name,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size
    )
