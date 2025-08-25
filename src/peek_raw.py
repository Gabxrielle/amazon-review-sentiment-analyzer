from huggingface_hub import list_repo_tree
repo = "McAuley-Lab/Amazon-Reviews-2023"
paths = [p.path for p in list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo="raw/review_categories") if p.path.endswith(".jsonl")]
print("Total on raw/:", len(paths))
for p in sorted(paths)[:40]:
    print("-", p)