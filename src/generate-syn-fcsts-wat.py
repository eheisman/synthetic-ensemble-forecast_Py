###
#  this is functionally the same as create_synthetic_forecasts_hec-wat-fra.py
# but designed to organize the inputs as a command line call for the HEC-WAT process
### 

import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
from datetime import datetime

from synfcst_wat.generate import SynFcstGenerator, WatCompute
from synfcst.model_params import ModelParams

import argparse

def main():
    # initialize generator
    # WatCompute object used to get the initial resources, move them into the right locations
    parser = argparse.ArgumentParser(prog="SyntheticForecasts",
                                 description="Generates synthetic forecasts",
                                 epilog="see ...")

    # 	cmdLine += [pythonExe, synFcstScript, fcstConfigFilename, modelParametersFilename]
    # remove "help" as these are optional args now?
    parser.add_argument("COMPUTE_OPTIONS")# , help="compute options that determines compute dimensions")
    parser.add_argument("MODEL_PARAMETERS") #, help="fitting parameters for generating forecasts")
    parser.add_argument("--debug")

    args = parser.parse_args()

    fcst_config = WatCompute(Path(args.COMPUTE_OPTIONS))
    wat_synfcst_dir = Path(fcst_config.watershed) / "synfcst"
    os.makedirs(Path(fcst_config.outDirectory), exist_ok=True)
    mod_params = ModelParams(Path(args.MODEL_PARAMETERS))
    generator = SynFcstGenerator(fcst_config, mod_params)

    #record complete time 
    now=datetime.now()
    print('gen start',now.strftime("%H:%M:%S"))

    # create forecasts
    generator.compute()

    now=datetime.now()
    print('gen end',now.strftime("%H:%M:%S"))


if __name__ == "__main__":
    sys.exit(main())
