import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import HuberLoss
from torch.optim import Adam
from homework.datasets.road_dataset import load_data
from homework.models import load_model, save_model



def train(model_name, num_epoch, lr, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    #print(f"Training {model_name} on {device}")
    #print(f"Requested batch size: {batch_size}")
    #print(f"[DEBUG] DataLoader batch_size = {batch_size}")

    transform_pipeline = ("state_only" if model_name in ["mlp_planner", "transformer_planner"] else "image_only")    
    # Load data
    dataloader = load_data(
      dataset_path="/content/homework4/drive_data/train",
      transform_pipeline=transform_pipeline,  # Load validation images in batches of 32, no random changes here because we are using this to test on how good th model is
      return_dataloader=True,
      batch_size=batch_size,
      shuffle=True,)



    # Load model
    model = load_model(model_name)
    model.to(device)

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = HuberLoss(delta=1.0)

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
            if model_name in ["mlp_planner", "transformer_planner"]:
              if track_left is None or track_right is None:
                print("Skipping batch with missing track data")
                continue
              
              track_left = track_left.to(device)
              track_right = track_right.to(device)
              inputs = {
                "track_left": track_left,
                "track_right": track_right,
        }
            elif model_name == "cnn_planner":
              if image is None:
                print("Skipping batch with missing image data")
                continue
              inputs = {
                "image": image.to(device),
        }
            else:
              raise ValueError(f"Unknown model: {model_name}")
            
            
            pred = model(**inputs)
            #print(f"track_left: {track_left.shape}, device: {track_left.device}")
            #print(f"track_right: {track_right.shape}, device: {track_right.device}")
            #print(f"Model is on device: {next(model.parameters()).device}")
            #print(f"pred.shape = {pred.shape}")

            
            # We are training a race car AI and every car (in a batch) sees a track
            # and gives 3 guesses about where to drive next like checkpoints
            # So if we have 64 cars in one roundm we have 192 guesses because
            # 64 * 3 guesses each = 192 guesses
            # Not all guesses are valid, some give shitty advice like driving 
            # off a cliff, so we use a mask to say which are good
            # So we are giving every car's guesses in 1 long line, and we flatten 
            # The mask in a long list of yes or no:
            # We say: only_good_guesses = guesses[mask], no shape mismatch, pytorch is happy
            pred_flat = pred.view(-1, 2)
            waypoints_flat = waypoints.view(-1, 2)
            mask_flat = mask.view(-1).bool()

            pred_masked = pred_flat[mask_flat]
            waypoints_masked = waypoints_flat[mask_flat]

            loss = loss_fn(pred_masked, waypoints_masked)

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
    parser.add_argument("--batch_size", type=int, default=64)


    args = parser.parse_args()
    train(
        model_name=args.model_name,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size
    )
