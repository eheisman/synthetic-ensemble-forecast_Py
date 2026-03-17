import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
from datetime import datetime

from synfcst_wat.generate import SynFcstGenerator, WatCompute
from synfcst.model_params import ModelParams

# initialize generator
# WatCompute object used to get the initial resources, move them into the right locations
fcst_config = WatCompute(Path("./tests/wat/resources/test-synfcst_config.json"))

# test config/setup
# create fake WAT model directory
os.makedirs(Path(fcst_config.watershed) / "synfcst", exist_ok=True)
wat_synfcst_dir = Path(fcst_config.watershed) / "synfcst"
shutil.copy("./data/ado/ADO_hefs_gefs_daily.npz", wat_synfcst_dir)
shutil.copy("./out/ADO/keysite=ADOC1/optimized-parameters_keysite=ADOC1_opt-pct=0.99.pkl", wat_synfcst_dir)
shutil.copy("./tests/wat/resources/prado_model_params.json", wat_synfcst_dir)

# copy lifecycle DSS into fake WAT lifecycle directory
os.makedirs(Path(fcst_config.outDirectory), exist_ok=True)
shutil.copy("./data/ado/HFO-FRA_50yr.dss", fcst_config.outDirectory)

mod_params = ModelParams(Path(wat_synfcst_dir / "prado_model_params.json")
generator = SynFcstGenerator(fcst_config, mod_params)

#record complete time 
now=datetime.now()
print('gen start',now.strftime("%H:%M:%S"))

# create forecasts
generator.compute()

now=datetime.now()
print('gen end',now.strftime("%H:%M:%S"))