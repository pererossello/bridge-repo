import os
import shutil

from numpyro import handlers

from ..forward_model.config import FMConfig
from ..forward_model.fmodel import FModel


class RecMaker:

    def __init__(self, fm_cfg: FMConfig, data):

        def tempered_model(data, T=1):
            with handlers.scale(scale=1.0 / T):
                FModel(fm_cfg).buil_model()(data)  # type: ignore

        self.model = tempered_model
        self.data = data

        return

    def set_up(self):

        return

    def _handle_dir(self, rec_dir, rec_name):
        self.main_dir = os.path.join(rec_dir, rec_name)
        if os.path.exists(self.main_dir):
            print("Directory\n" + f"{self.main_dir}\n" + "already exists")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response == "y":
                shutil.rmtree(self.main_dir)
                print("Directory deleted.")
            else:
                raise ValueError("Operation aborted.")

        print("Creating directory\n" + f"{self.main_dir}\n")
        os.makedirs(self.main_dir)

        self.samples_dir = os.path.join(self.main_dir, "00CH")
        os.makedirs(self.samples_dir)

    def _handle_subdirs(self):
        self.plots_dir = os.path.join(self.samples_dir, "figures")
        self.results_dir = os.path.join(self.samples_dir, "results")

        os.makedirs(self.plots_dir)
        os.makedirs(self.results_dir)
