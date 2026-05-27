from hec.hecmath import TimeSeriesMath
from com.rma.io import DssFileManagerImpl

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

nStepsPerDay = {
	"1DAY": 1,
	"12HOUR": 2,
	"6HOUR": 4,
	"3HOUR": 8,
	"1HOUR": 24,
	"30MIN": 48,
	"15MIN": 24*4,
	"10MIN": 24*6,
	"5MIN": 24*12,
	"1MIN": 24*60
}

def computeAlternative(currentAlternative, computeOptions):
	currentAlternative.addComputeMessage("Computing ScriptingAlternative:" + currentAlternative.getName() )
	#write_example_dl(currentAlternative)
	for odl in currentAlternative.getOutputDataLocations():
		locName = odl.getName()
		paramName = odl.getParameter()

		# if there are commands, process them into a dictionary
		# use anything after a colon as the commands
		# e.g. "location1-unreg:FMA=72" will compute the 72hr unreg flow forward moving average.
		settings = dict()
		if ":" in locName:
			locName, commands = locName.split(":")
			for cmd in commands.split(" "):
				if "=" in cmd:
					k,v = cmd.split("=")
					# store as upper case, don't do anything with the value, as it could be anything.
					settings[k.upper()] = v
	
		# read timeseries
		#idl = currentAlternative.getInputDataLocation(locName, paramName)
		inTSM = TimeSeriesMath(currentAlternative.getTimeSeriesForInputDataLocation(locName,paramName))
		# convert to timestep (daily in this case)
		tsm = inTSM.transformTimeSeries(currentAlternative.getTimeStep(), "", "AVE")
		
		# compute forward moving average on N timesteps
		if "FMA" in settings.keys():
			nAvg = int(settings["FMA"])
			# true, false to get only valid values and whole days
			tsm = tsm.forwardMovingAverage(nAvg, True, True)
			# assumes daily in the next step, shifting to line up with fcst values
			tsm = tsm.shiftInTime("-%d%s" % (nAvg-1, "D"))
		if "VOL" in settings.keys():
			scriptingStep = currentAlternative.getTimeStep().upper()
			tsm = tsm.multiply(nAvg*24/12.1/nStepsPerDay[scriptingStep]) # cfs-days to acre-feet
			tsm.setUnits("ACRE-FT")  # consistent with FIRO_TSEnsembles for now
			tsm.setParameterPart("VOL-%d DAY" % nAvg) # would be nice, OVs don't work though
		# tsm.setLocation(odl.getName())
  		# tsm.setParameterPart(odl.getParameter())  #NEW!  This is important for the OutputVariables to work correctly.
		currentAlternative.writeTimeSeries(tsm.getData())
		currentAlternative.addComputeMessage("\tsuccesfully computed for output %s:%s" % (odl.getName(), odl.getParameter()))
	return True
