import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('./src'))
from pathlib import Path
from datetime import datetime

from synfcst_wat.generate import SynFcstGenerator
from synfcst.model_params import ModelParams

# initialize generator
generator = SynFcstGenerator(Path("./tests/wat/resources/test-synfcst_config.json"))

# test config/setup
# copy DSS into fake WAT lifecycle directory
os.makedirs(Path(generator.compute_options.outDirectory), exist_ok=True)
shutil.copy("./data/ado/HFO-FRA_50yr.dss", generator.compute_options.outDirectory)
# create fake WAT model directory
os.makedirs(Path(generator.compute_options.watershed) / "synfcst", exist_ok=True)
wat_synfcst_dir = Path(generator.compute_options.watershed) / "synfcst"
shutil.copy("./data/ado/ADO_hefs_gefs_daily.npz", wat_synfcst_dir)
shutil.copy("./out/ADO/keysite=ADOC1/optimized-parameters_keysite=ADOC1_opt-pct=0.99.pkl", wat_synfcst_dir)
shutil.copy("./tests/wat/resources/prado_model_params.json", wat_synfcst_dir)

modParams = ModelParams(Path(wat_synfcst_dir) / "prado_model_params.json")

#record complete time 
now=datetime.now()
print('gen start',now.strftime("%H:%M:%S"))

# create forecasts
generator.compute(modParams)

now=datetime.now()
print('gen end',now.strftime("%H:%M:%S"))