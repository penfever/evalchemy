#!/usr/bin/env python3
import os
import argparse
import sys
from huggingface_hub import HfApi, create_repo, CommitOperationAdd, RepositoryNotFoundError

def upload_directory_to_hf(directory_path, repo_name, repo_type, token=None):
    """
    Upload all files from a directory to a private repository on Hugging Face Hub.
    If the repository already exists, it will add the files as a new commit.
    
    Args:
        directory_path: Path to the directory containing files to upload
        repo_name: Name for the Hugging Face repository
        repo_type: Type of repository (model, dataset, space)
        token: Hugging Face API token (if None, will look for environment variables)
    """
    # Validate directory path
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    # Validate repo type
    valid_repo_types = ["model", "dataset", "space"]
    if repo_type not in valid_repo_types:
        print(f"Error: Repository type '{repo_type}' not valid. Choose from: {', '.join(valid_repo_types)}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Get authentication token
    if token is None:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token is None:
            print("Error: Hugging Face token not provided and neither HUGGING_FACE_HUB_TOKEN nor HF_TOKEN environment variables are set", 
                  file=sys.stderr)
            sys.exit(1)
    
    api = HfApi(token=token)
    
    try:
        # Check if repository exists
        repo_exists = True
        try:
            api.repo_info(repo_id=repo_name, repo_type=repo_type)
            print(f"Repository {repo_name} already exists, will add files as a new commit")
        except RepositoryNotFoundError:
            repo_exists = False
            print(f"Creating new private {repo_type} repository: {repo_name}")
            create_repo(repo_id=repo_name, token=token, private=True, repo_type=repo_type)
        
        # List all files in the directory (including in subdirectories)
        operations = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path for the file in the repo
                relative_path = os.path.relpath(file_path, directory_path)
                
                # Read file content
                with open(file_path, "rb") as f:
                    content = f.read()
                
                # Add file to operations list
                operations.append(CommitOperationAdd(
                    path_in_repo=relative_path,
                    path_or_fileobj=content
                ))
                print(f"Adding file: {relative_path}")
        
        # Upload all files in a single commit
        if operations:
            commit_message = "Update existing files" if repo_exists else "Initial upload of all files"
            print(f"Uploading {len(operations)} files to {repo_name}...")
            api.create_commit(
                repo_id=repo_name,
                operations=operations,
                commit_message=commit_message,
                repo_type=repo_type
            )
            print(f"Successfully uploaded all files to https://huggingface.co/{repo_name}")
        else:
            print("No files found in the directory to upload")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload directory contents to a private Hugging Face repository")
    parser.add_argument("directory", help="Path to directory containing files to upload")
    parser.add_argument("--repo-name", help="Name for the Hugging Face repository", required=True)
    parser.add_argument("--repo-type", 
                        help="Type of repository (model, dataset, space)", 
                        default="model",
                        choices=["model", "dataset", "space"])
    parser.add_argument("--token", help="Hugging Face API token (optional if environment variables are set)")
    
    args = parser.parse_args()
    
    # Upload directory to Hugging Face
    upload_directory_to_hf(args.directory, args.repo_name, args.repo_type, args.token)

if __name__ == "__main__":
    main()