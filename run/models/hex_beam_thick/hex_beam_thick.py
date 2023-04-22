## imports
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

from odbAccess import openOdb

import pickle
import numpy as np
import sys

young_modulus = 70000.0
l = 1.0 
h = 1.5*l # characteristic cell size
t = 0.06 # middle section thickness

young_modulus = 70000.0
nu = 0.3 # Poisson's ratio 


remote_strain = -0.001

y_disp = remote_strain*2*h

def create_sim(model_name,job_name,parametrization,save_cae=False):
    
    eta = parametrization['eta']
    xi = parametrization['xi']
    
    
    #t_h = t*(2*eta + xi*eta)/(2+xi*eta)
    t_h = t*eta/(1-xi+xi*eta)
    
           
    model = mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)
    
    # Delete 'Model-1' if it's already here
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']
    
    
    
    # Sketch 
    s = model.ConstrainedSketch(name='__profile__', sheetSize=5.0)
    
    # Generate hexagon of given wall length a
    def generate_hex(a):    
        h = a*(np.sqrt(3)/2) # height
        
        return [(0,a),(h,a/2),(h,-a/2),(0,-a),(-h,-a/2),(-h,a/2)] # boundary nodes
    
    
    
    i_points = generate_hex(l) # Generate the inner hexagon points
    o_points = generate_hex(1.5*l) # Generate the outer hexagon
    
    for idx in range(6): # iterate through nodes and sketch lines
      p1 = i_points[idx]
      p2 = i_points[(idx+1)%6]
      p3 = o_points[idx]
      
      s.Line(point1=p1, point2=p2) # Hexagon wall
      s.Line(point1=p1, point2=p3) # Outter appendices
      
      
    part = model.Part(dimensionality=TWO_D_PLANAR, name='HEX_BEAM', type=DEFORMABLE_BODY)
    part.BaseWire(sketch=s)
    del s
            
    # Define the material
    material_name = 'ALUMINIUM'
    material = model.Material(name=material_name)
    material.Elastic(table=((young_modulus, nu), ))
    
    # Extract inner and outer edges for partitioning
    pick_point = lambda p1,p2 : (p1[0]+0.99*(p2[0]- p1[0]),p1[1] + 0.99*(p2[1]-p1[1]))
    
    for idx in range(6):
          p1 = i_points[idx]
          p2 = i_points[(idx+1)%6]
          p3 = o_points[idx]
          
          i_pick = pick_point(p1,p2)+(0,)
          o_pick = pick_point(p1,p3)+(0,)
          
          # L - middle part length (xi*l)
          # l - entire length
          
          i_edge = part.edges.findAt((i_pick,))
          part.PartitionEdgeByParam(edges=i_edge,parameter = (1-xi)/2) # L/l
          
          i_edge = part.edges.findAt((i_pick,))
          part.PartitionEdgeByParam(edges=i_edge,parameter = (2*xi)/(1+xi)) # (l-2L)/(l-L)
          
          o_edge = part.edges.findAt((o_pick,))      
          part.PartitionEdgeByParam(edges=o_edge,parameter = (1-xi)) # 2L/l
    
    ## Selecting sets      
    # Select the entire unit cell
    bound = 2*l
    unit_cell_set = part.Set(edges=part.edges.getByBoundingBox(-bound, -bound,0,bound,bound,0), name='UNIT_CELL')     
    
    # Select the joints
    joint_r  = 1.001*l/(xi+2)
    joints = [part.edges.getByBoundingSphere(i_points[i]+(0,),joint_r) for i in range(6)]
    joints_set = part.Set(edges=tuple(joints), name='JOINTS')
    
    # Select beams
    beams_set = part.SetByBoolean(name = 'BEAMS', operation = DIFFERENCE, sets = (unit_cell_set,joints_set) )
    
    
    # Assign section orientation
    part.assignBeamSectionOrientation(method=N1_COSINES, n1=(0.0, 0.0, -1.0), region = unit_cell_set)
    
    # Create profiles
    model.RectangularProfile(a=1.0, b=t_h, name='MID_PROFILE')
    model.RectangularProfile(a=1.0, b=t_h/eta, name='END_PROFILE')
    
    # Create sections
    model.BeamSection(consistentMassMatrix=False, 
        integration=DURING_ANALYSIS, material=material_name, name='MID_SECTION', 
        poissonRatio=0.0, profile='MID_PROFILE', temperatureVar=LINEAR)
    
    # Create end section
    model.BeamSection(consistentMassMatrix=False, 
        integration=DURING_ANALYSIS, material=material_name, name='END_SECTION', 
        poissonRatio=0.0, profile='END_PROFILE', temperatureVar=LINEAR)
    
    # Middle section
    part.SectionAssignment(region=beams_set, sectionName='MID_SECTION', thicknessAssignment=FROM_SECTION)
    # # End section 
    part.SectionAssignment(region=joints_set, sectionName='END_SECTION', thicknessAssignment=FROM_SECTION)
    
    # Create instance
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    instance = a.Instance(dependent=ON, name = 'INST', part=part)
    
    # Create step
    model.StaticStep(initialInc=0.01, maxInc=0.01,maxNumInc=1000, minInc=0.01, name='Step-1', previous='Initial')
    
    # Mesh on part
    part.seedPart(deviationFactor=0.1, minSizeFactor=0.01, size=0.01)
    part.generateMesh()
    
    
    a.regenerate()
    
    # Create a reference point
    ref_point=a.ReferencePoint(point=(bound, -bound, 0))
    ref_id = ref_point.id
    ref_point= a.referencePoints[ref_id]
    
    ref_set = a.Set(name='REF', referencePoints=(ref_point, ))
    
    # Put each node into set
    for idx in range(6): # iterate through nodes
      p = o_points[idx]
      p_node = instance.nodes.getClosest(coordinates=(p+(0,)))
      p_node = MeshNodeArray((p_node,)) # Set takes meshnodearray
      a.Set(name='n{}'.format(idx+1), nodes= (p_node,))
    
    # Equation BC
    model.Equation(name='Constraint-1', terms=((1.0, 'n4', 1), (-1.0, 'n1', 1))) # Correct
    model.Equation(name='Constraint-2', terms=((1.0, 'n5', 1), (-1.0, 'n2', 1), (1.0, 'REF', 1))) # Correct
    model.Equation(name='Constraint-3', terms=((1.0, 'n6', 1), (-1.0, 'n3', 1), (1.0, 'REF', 1))) # Correct 
    model.Equation(name='Constraint-4', terms=((1.0, 'n4', 2), (-1.0, 'n1', 2), (1.0, 'REF', 2))) # Correct
    model.Equation(name='Constraint-5', terms=((1.0, 'n5', 2), (-1.0, 'n2', 2), (0.5, 'REF', 2))) # Correct
    model.Equation(name='Constraint-6', terms=((1.0, 'n6', 2), (-1.0, 'n3', 2), (-0.5, 'REF', 2))) # Correct
    model.Equation(name='Constraint-7', terms=((1.0, 'n4', 6), (-1.0, 'n1', 6))) # Correct
    model.Equation(name='Constraint-8', terms=((1.0, 'n5', 6), (-1.0, 'n2', 6))) # Correct
    model.Equation(name='Constraint-9', terms=((1.0, 'n6', 6), (-1.0, 'n3', 6))) # Correct    
    


    model.DisplacementBC(amplitude=UNSET, createStepName=
    'Step-1', distributionType=UNIFORM, name='BC-1', region=
    ref_set, u1=UNSET, u2=
    y_disp, ur3=UNSET)
            
            
    
    modelJob = mdb.Job(model=model_name, name=job_name)      
    modelJob.submit(consistencyChecking=ON)
    modelJob.waitForCompletion()
    
    if save_cae:
        mdb.saveAs(job_name + '.cae')

def post_process(job_name, param_vector):
    
    # odb opening    
    odb_name = '{}.odb'.format(job_name)
    
    try:
        odb =  session.odbs[odb_name]
    except:    
        odb = session.openOdb(name=odb_name)
            
    # initialize an empty dictionary
    sim_results = {}
    
    
    # field outputs
    step = odb.steps[odb.steps.keys()[-1]] # last step
    frame = step.frames[-1]  # last frame in the last step
    RF = frame.fieldOutputs['RF'] # reaction forces
    U = frame.fieldOutputs['U']

    # extract the data for reference node
    RP = odb.rootAssembly.nodeSets['REF'] # reference point set
    RF_at_RP= RF.getSubset(region=RP)
    U_at_RP = U.getSubset(region=RP)
    
    force = RF_at_RP.values[0].data[1]
    
    rel_density = (2.0/np.sqrt(3))*(t/l)*(1.0-(t/l)/(2.0*np.sqrt(3)))
    
    u1 = U_at_RP.values[0].data[0]
    u2 = U_at_RP.values[0].data[1]
    
    scaling_stiffness = (1.5*(rel_density)**3.0)*young_modulus
            
    e11 = u1/(sqrt(3)*h)
    e22 = u2/(2*h)
    # sim_results['v'] = -sim_results['e11']/sim_results['e22']

    s22 = force/(1.5*sqrt(3)*l)
    
    uniform_stiff = scaling_stiffness
    thickened_stiff = s22/e22
    
    sim_results['stiff_ratio'] = thickened_stiff/uniform_stiff 
    
    odb.close()    
    
    pickle_name = job_name +'_results.pkl'
    
    f = open(pickle_name,'wb')
    pickle.dump(sim_results,f)
    f.close()
    

