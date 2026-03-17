# WAT 
- refactor into WAT generator as a module (in progress)
- connect to Prado HFO FRA compute (next)
- add deterministic and stochastic POR capability
- update WAT jython script to call Python not R
- WAT jython script: write per-aver data for forecasts to DSS
- WAT jython script: pass event DSS file into script
- WAT jython script:  identify datalocations for script
- WAT jython script: compute-per-lifecycle vs compute-per-event, pass via config
- separate data, synfcst code, python directory
    - data goes into WAT watershed as the "model alternative"
    - synfcst code and python directory go into users %appdata% as a "wat plugin"

## DONE
- DSS: limit DSS messages through `zset msglvl` call.
- writes to "test watershed" for WAT compute to use .json file
- put WAT scripting alt into repo (DONE) - but needs updating!

# overall
- modularize the synthetic generation - started
- add tests for WAT behavior to streamline development/refactoring  - in progress