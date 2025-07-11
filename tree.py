#tree.py
import os

def print_tree(dir_path, prefix=''):
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path) and item not in ['__pycache__', '.git', 'env', 'venv']:
            print(prefix + ' ' + item)
            print_tree(path, prefix + '    ')
        elif os.path.isfile(path):
            print(prefix + ' ' + item)

print_tree('.')
