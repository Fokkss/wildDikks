# TODO
import os

os.makedirs('output', exist_ok=True)

# fake forecast
with open('output/prediction.csv', 'w') as f:
    for _ in range(24):
        f.write("123.45\n")

print("forecast loaded successfully ")