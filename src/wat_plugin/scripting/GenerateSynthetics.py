import os
import json
import subprocess
from java.lang import System
from hec.script import Constants
from hec.hecmath import TimeSeriesMath

# new setting to run once per lifecycle.
PER_LIFECYLCE_COMPUTE = False


#scriptConfigFilename = "synForecasts/forecastConfig.json"
def generateSynthetics(opts, configFile, synFcstScript=None, args=[], shell=False):
	watershedDir = opts.getRunDirectory().split("runs")[0]
	# where is the plugin code?
	synFcstPluginDir = os.path.join(os.getenv("APPDATA"), "HEC", "HEC-WAT", "Plugins", "SynFcst")
	if synFcstScript is None:
		synFcstScript = os.path.join(synFcstPluginDir, "src", "generate-syn-fcsts-wat.py")

	# set location relative to the python venv used
	pythonDir = os.path.join(synFcstPluginDir, "python/")
	pythonExe = os.path.join(pythonDir, "Scripts", "python.exe")

	# model parameters
	modelParametersFilename = os.path.join(watershedDir, "synFcst", "model_params.json")

	# to configure:
	# create python venv, add dependencies with pip, including synfcst module from files

	# set working directory to WAT simulation directory.
	# SOMEDAY BETTER
	#os.chdir(getOutputDir(opts))
	# For now, workingdir = appdata so we can load the module
	os.chdir(synFcstPluginDir)
	# call Python
	cmdLine = []
	cmdLine = [pythonExe, synFcstScript, configFile, modelParametersFilename]
	cmdLine += args
	# TODO: this could be cleaner than it was, as it is, I'm going to reuse the R script version.
	# I need to clean this up, I cannot make it launch a separate shell to monitor the process
	# this appears to not work with spaces in the arguments
	if shell: cmdLine = ["start", "cmd", "/k ^\n", " ".join(cmdLine)] #+ cmdLine
	# quotations around the program being called can make a big difference here.
	subprocess.call(" ".join(cmdLine), shell=shell) # This works if shell=False
	# TODO: use Popen class to facilitate communication back to WAT.
	#p = subprocess.Popen(cmdLine, shell=True) 
	#p.wait()
	return " ".join(cmdLine)

def getOutputDir(opts):
	d = os.path.dirname(opts.getRunDirectory())
	d = d.replace("Scripting", "")
	if PER_LIFECYLCE_COMPUTE:
		d = d.replace("event 1", "")
	if not os.path.exists(d):
		os.mkdir(d)
	return d

def stringWrap(s, force=False):
	#quotes = ["'", "\"",]
	#if not force and (s[0] in quotes) and (s[-1] in quotes):
	#	return s
	#else:
	return "\"%s\"" % s

def writeScriptConfig(alt, opts):
	## Writes out a configuration file for synfcst script to reference
	config = dict()
	config["LifecycleCompute"] = PER_LIFECYLCE_COMPUTE
	config["SynFcstTimestep"] =  alt.getTimeStep()
	# create run time window details
	# Currently this is per event, not for the lifecycle.  Need to add switch
	# TODO: these don't get used much at all due to the way the script uses valid data to determine when to run.
	rtwDict = dict()
	if not PER_LIFECYLCE_COMPUTE:
		rtw = opts.getRunTimeWindow()
		rtwDict["Start Time"] = rtw.getStartTimeString()
		rtwDict["End Time"] = rtw.getEndTimeString()
	config["TimeWindow"] = rtwDict

	## Save realization and event seeds
	if opts.isFrmCompute():
		config["ComputeType"] = "FRM"
		eventList = opts.getEventList()
		# not technically seeds, but these can be used to generate a seed for the R script
		randomDict = dict()
		randomDict["Event Random"] = opts.getEventRandom()
		randomDict["Realization Random"] = opts.getRealizationRandom()
		randomDict["Lifecycle Random"] = opts.getLifeCycleRandom()
		config["Randoms"] = randomDict
		# set indicies for where we are working.
		indexDict = dict()
		indexDict["Event Number"] = opts.getCurrentEventNumber()
		indexDict["Lifecycle Number"] = opts.getCurrentLifecycleNumber()
		indexDict["Realization Number"] = opts.getCurrentRealizationNumber()
		indexDict["Events Per Lifecycle"] = len(eventList)
		config["Indices"] = indexDict
	else:
		config["ComputeType"] = "Deterministic"

	# get DSS output data:
	outputDict = dict()
	outputDict["Run Directory"] = opts.getRunDirectory()
	watershedDir = opts.getRunDirectory().split("runs")[0]
	outputDict["Watershed Directory"] = watershedDir
	outputDict["Simulation Name"] = opts.getSimulationName()
	outputDict["Out Directory"] = getOutputDir(opts)
	outputDict["DSS File"] = opts.getDssFilename()
	outputDict["F Part"] = opts.getFpart()
	config["Outputs"] = outputDict

	# create list of locations mapped in
	locations = alt.getInputDataLocations()
	config["Locations"] = list()
	for loc in locations:
		locDict = dict()
		locDict["name"] = loc.getName()
		locDict["param"] = loc.getParameter()
		locPath = loc.getDssPath()
		locDict["dss_pathname"] = locPath.replace("alt:ap:Scripting-SynFcst", opts.getFpart())
		# this is the input location from the HS model 
		#loc.getLinkedToLocation().getDssPath()
		alt.addComputeMessage("linked input \"%s/%s\" to %s" % (locDict["name"], locDict["param"], locDict["dss_pathname"]))
		config["Locations"].append(locDict)

	# write to file
	d = getOutputDir(opts)
	#if not PER_LIFECYLCE_COMPUTE:
	#	os.path.join(d, "event %d" % opts.getCurrentEventNumber())
	configFilename = os.path.join(d, "SynFcstConfig.json") 
	with open(configFilename, 'w') as out:
		out.write(json.dumps(config))
	
	return stringWrap(configFilename)
	

def createDailyAverageForForecasts(alt, opts):
	"""
	"""
	# TODO: do this for all events! otherwise we have to generate forecasts one at a time.
	for inputLoc in alt.getInputDataLocations():
		inputTSM = TimeSeriesMath(alt.loadTimeSeries(inputLoc))
		# see https://www.hec.usace.army.mil/confluence/dssdocs/dssvueum/scripting/math-functions
		# Computes daily average anchored at midnight end-of-day.  offsetString can be used to shift the calculation, but leaving blank.
		offsetString = ""
		dailyTSM = inputTSM.transformTimeSeries(alt.getTimeStep(), offsetString, "AVE")
		# this ensures the output data location is used.
		outputLoc = currentAlternative.getOutputDataLocation(inputLoc.getName(), inputLoc.getParameter())
		dailyTSM.setPathname(outputLoc.getDssPath())
		alt.writeTimeSeries(dailyTSM.getData())

##
#
# computeAlternative function is called when the ScriptingAlternative is computed.
# Arguments:
#   currentAlternative - the ScriptingAlternative. hec2.wat.plugin.java.impl.scripting.model.ScriptPluginAlt
#   computeOptions     - the compute options.  hec.wat.model.ComputeOptions
#
# return True if the script was successful, False if not.
# no explicit return will be treated as a successful return
#
##
def computeAlternative(currentAlternative, computeOptions):
	currentAlternative.addComputeMessage("Computing ScriptingAlternative:" + currentAlternative.getName())
	# if PER_LIFECYLCE_COMPUTE and computeOptions.getCurrentEventNumber() > 1:
	#	currentAlternative.addComputeMessage("This already computed in event 1. Not recomputing.")
	#	return True
	# write configuration for script - tells it compute timewindow and locations
	configFile = writeScriptConfig(currentAlternative, computeOptions)

	# write timeseries
	#dataFile = writeTsCSV(currentAlternative, computeOptions, outTimestep="1DAY", timestampColumnName="GMT")
	
	# create daily average data
	createDailyAverageForForecasts(currentAlternative, computeOptions)

	# run R compute function here 
	#rScriptFile = None #r"synForecasts\wat_launcher.R"
	#currentAlternative.addComputeMessage(callR(rScriptFile, computeOptions, [configFile, dataFile], relativeScript=True))
	resultsMessage = '' #generateSynthetics(computeOptions, configFile)
	currentAlternative.addComputeMessage(resultsMessage)

	return True
