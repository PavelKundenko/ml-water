import os

# Get the absolute path to the project's root directory
PROJECT_ROOT = '/Users/pavlo.kundenko/PhD/Projects/ml-water/src'


# Function to resolve a file path relative to the project root
def resolve_path(relative_path, create_dir=False):
  resolved_path = os.path.join(PROJECT_ROOT, relative_path)

  # Check if the path is a directory
  if not os.path.exists(resolved_path) and create_dir:
    os.makedirs(resolved_path, exist_ok=True)  # Create the directory

  return resolved_path

