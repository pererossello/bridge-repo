import os
import shutil


def make_and_check_dir(a_dir, overwrite=True):
    if os.path.exists(a_dir):
        print("Directory\n" + f"{a_dir}\n" + "already exists")
        response = input("Overwrite? (y/n): ").strip().lower()
        if overwrite == False:
            return
        if response == "y":
            shutil.rmtree(a_dir)
            print("Directory deleted.")
        else:
            raise ValueError("Operation aborted.")
    print("Creating directory\n" + f"{a_dir}\n")
    os.makedirs(a_dir)
