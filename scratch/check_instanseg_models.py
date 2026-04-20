import instanseg
from instanseg.utils.utils import get_model_index

try:
    index = get_model_index()
    print("Available models:")
    for model_name in index.keys():
        print(f" - {model_name}")
except Exception as e:
    print(f"Error: {e}")
