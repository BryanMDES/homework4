import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
    best_loss = float("inf")
    best_model_state = None
    best_lat_error = float("inf")
    best_lon_error = float("inf")
  

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
            mask_flat = mask.view(-1).bool() # Flatten the mask

            # Apply the mask to select only valid waypoints
            pred_masked = pred_flat[mask_flat]
            waypoints_masked = waypoints_flat[mask_flat]
            lat_loss = F.smooth_l1_loss(pred_masked[:, 0], waypoints_masked[:, 0])
            lon_loss = F.smooth_l1_loss(pred_masked[:, 1], waypoints_masked[:, 1])
            loss = 3.0 * lat_loss + lon_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # total loss for this epoch to monitor
            total_loss += loss.item()


        # Evaluate metrics at end of epoch
        model.eval()
        with torch.no_grad():
          lat_errors = []
          lon_errors = []

          for batch in dataloader:
            track_left = batch.get("track_left", None)
            track_right = batch.get("track_right", None)
            image = batch.get("image", None)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            if model_name in ["mlp_planner", "transformer_planner"]:
              if track_left is None or track_right is None:
                  continue
              inputs = {
                  "track_left": track_left.to(device),
                  "track_right": track_right.to(device),
              }
            elif model_name == "cnn_planner":
              if image is None:
                 continue
              inputs = {"image": image.to(device)}
            else:
              continue

            pred = model(**inputs) # Providing predictions
            pred = pred.view(-1, 2) #Flatten
            waypoints = waypoints.view(-1, 2)
            mask = mask.view(-1).bool()

            pred_masked = pred[mask] # Providing predictoins that actually matter
            waypoints_masked = waypoints[mask] # Same for the targets

            lat_error = F.l1_loss(pred_masked[:, 0], waypoints_masked[:, 0])
            lon_error = F.l1_loss(pred_masked[:, 1], waypoints_masked[:, 1])

            lat_errors.append(lat_error.item())
            lon_errors.append(lon_error.item())

        avg_lat = sum(lat_errors) / len(lat_errors)
        avg_lon = sum(lon_errors) / len(lon_errors)

        if avg_lat < best_lat_error and avg_lon < best_lon_error: # Both need to be better
          best_lat_error = avg_lat
          best_lon_error = avg_lon
          best_model_state = model.state_dict() # Save the best model
          print(f"Saved best model at epoch {epoch + 1} | lat: {avg_lat:.4f}, lon: {avg_lon:.4f}")

    if best_model_state is not None:
      model.load_state_dict(best_model_state)
      path = save_model(model) # Save it forever
      print(f"Best model saved with lat={best_lat_error:.4f}, lon={best_lon_error:.4f}")


    
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
