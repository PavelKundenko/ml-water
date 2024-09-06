import os

# Get the absolute path to the project's root directory
PROJECT_ROOT = '/Users/pavlo.kundenko/PhD/Projects/ml-water/src'


# Function to resolve a file path relative to the project root
def resolve_path(relative_path):
  resolved_path = os.path.join(PROJECT_ROOT, relative_path)

  # Check if the resolved path exists
  if not os.path.exists(resolved_path):
    raise FileNotFoundError(f"Resolved path does not exist: {resolved_path}")

  return resolved_path

