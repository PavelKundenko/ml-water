import os

# Get the absolute path to the project's root directory
PROJECT_ROOT = '/Users/pavlo.kundenko/PhD/Projects/ml-water/src'


# Function to resolve a file path relative to the project root
def resolve_path(relative_path):
  return os.path.join(PROJECT_ROOT, relative_path)

