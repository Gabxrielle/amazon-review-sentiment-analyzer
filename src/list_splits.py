from huggingface_hub import list_repo_tree
items = list_repo_tree(repo_id="McAuley-Lab/Amazon-Reviews-2023", repo_type="dataset", path_in_repo="raw")
tops = sorted(set(p.path.split("/")[0] for p in items))
print("Top-level dirs:", tops)