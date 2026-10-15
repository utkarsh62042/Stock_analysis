import os
import shutil
import stat

def _force_remove(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def delete_pycache(root_dir="."):
    deleted = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path, onexc=_force_remove)
            deleted.append(pycache_path)
            dirnames.remove("__pycache__")  # don't recurse into it

    if deleted:
        print(f"Deleted {len(deleted)} __pycache__ folder(s):")
        for path in deleted:
            print(f"  ✓ {path}")
    else:
        print("No __pycache__ folders found.")

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    delete_pycache(root)