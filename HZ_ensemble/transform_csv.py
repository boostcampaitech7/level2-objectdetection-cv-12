import pandas as pd

pred_df = pd.read_csv('/data/ephemeral/home/baseline/HZ_ensemble/val_inf/dino_val_inf.csv')

pred_df['image_id'] = "train/" + pred_df['image_id']
pred_df.to_csv('real.csv')