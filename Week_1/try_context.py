import os

predict_base_dir = "context"

for fname in os.listdir(predict_base_dir):
    print(fname)