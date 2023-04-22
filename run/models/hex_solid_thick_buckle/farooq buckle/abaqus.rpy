# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2018 replay file
# Internal Version: 2017_11_07-17.21.41 127140
# Run by kuszczak_i on Wed Mar 23 15:52:50 2022
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=222.166656494141, 
    height=132.701400756836)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
openMdb('BU005.cae')
#: Warning: D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\run\models\hex_solid_thick_buckle\farooq buckle\BU005.cae is being open for read only. No model changes will be made.
#: The model database "D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\run\models\hex_solid_thick_buckle\farooq buckle\BU005.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['Model-1'].parts['part2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
a = mdb.models['Model-1'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=20.8943, 
    farPlane=26.6277, width=23.4109, height=12.7795, viewOffsetX=1.60767, 
    viewOffsetY=0.488075)
mdb.save()
#* IOError: 
#* D:/Temp/kuszczak_i/Studentship/bayes-opt-for-abaqus/run/models/hex_solid_thick_buckle/farooq 
#* buckle/BU005.cae: Permission denied
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF, adaptiveMeshConstraints=ON)
execfile(
    'D:/Temp/kuszczak_i/Studentship/bayes-opt-for-abaqus/run/models/hex_solid_thick_buckle/hex_solid_thick_buckle.py', 
    __main__.__dict__)
#: D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\run\catia\hex_solid_thick_bucle.stp
#: The model "hex_solid_thick_buckle" has been created.
#: Part 'HEX_JT_BUCKLE' contains valid geometry and topology.
#: Part 'HEX_JT_BUCKLE' is a shell part(5 shell faces, 416 edges, 402 vertices).
#* TypeError: unsupported operand type(s) for +: 'int' and 'tuple'
#* File 
#* "D:/Temp/kuszczak_i/Studentship/bayes-opt-for-abaqus/run/models/hex_solid_thick_buckle/hex_solid_thick_buckle.py", 
#* line 340, in <module>
#*     create_sim(model_name,job_name,parametrization,save_cae=False)
#* File 
#* "D:/Temp/kuszczak_i/Studentship/bayes-opt-for-abaqus/run/models/hex_solid_thick_buckle/hex_solid_thick_buckle.py", 
#* line 192, in create_sim
#*     p_ref =a.ReferencePoint(point=(p+(0,)))
a = mdb.models['hex_solid_thick_buckle'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
