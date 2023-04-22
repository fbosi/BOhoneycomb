# Abaqus imports
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
import os
import random 
import numpy as np
from win32com.client import Dispatch

l = 1.0 
h = 1.5*l # characteristic cell size in y direction (vertical span)
H = 0.5*np.sqrt(3)*l # characteristic cell size in x direction (horizontal span)

relative_density = 0.15 # imposed relative density
t_uni = -np.sqrt(3)*(np.sqrt(1-relative_density)-1)*l # middle section thickness for the given relative density(uniform)

## Material properties
young_modulus = 7000.0 #MPa
yield_strength = 130.0
nu = 0.3 # Poisson's ratio 


remote_strain = -0.05# Set to more than 1% to capture plastic effects
y_disp = 2*h*remote_strain # 

def create_sim(model_name,job_name,parametrization,save_cae=False):
    
    eta = parametrization['eta']
    xi = parametrization['xi']
    
    '''CATIA PART'''
    # Connecting to windows COM
    CATIA = Dispatch('CATIA.Application')
    # optional CATIA visibility
    CATIA.Visible = True
    CATIA.DisplayFileAlerts = False
    
    partDocument = CATIA.ActiveDocument
    part = partDocument.Part
    
    parameters = part.Parameters
    
    length = parameters.Item('l')
    thickness = parameters.Item('t_uniform')
    
    # Scaling is needed as CATIA operates in mm
    length.Value = l*1e3
    thickness.Value = t_uni*1e3
    xi_param = parameters.Item('xi')
    eta_param = parameters.Item('eta')
    try:
        xi_param.Value = xi
        eta_param.Value = eta
        partDocument.Part.Update()
    except Exception:
        print('The trial parametrization is incorrect')
        
    catia_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),'..','catia','%s.stp')%model_name)
    # hard code this for now cause it is problematic
    catia_dir = r'D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\run\catia\hex_solid_thick_bucle.stp'
    
    # This should go up one directory and then to catia folder to make sense
    partDocument.ExportData(catia_dir, 'stp')
    
    '''ABAQUS PART'''  
    model = mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)
    
    # Delete 'Model-1' if it's already here
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']   
        
    # Import the step file
    
    mdb.openStep(catia_dir, scaleFromFile=OFF)
    
    sketch_name = 'unit_cell_hex_thickened'
    part_name = 'HEX_JT_BUCKLE'
    part = model.PartFromGeometryFile(combine=False,
                                      dimensionality=TWO_D_PLANAR,
                                      geometryFile=mdb.acis,
                                      name=part_name,
                                      scale=0.001,
                                      type=DEFORMABLE_BODY)
    
    # This bit is important as it ensures correct meshing of the part
    part.ConvertToPrecise(method=RECOMPUTE_GEOMETRY)
    part.checkGeometry()
    
    # Define the material
    material_name = 'ALUMINIUM'
    material = model.Material(name=material_name)
    material.Elastic(table=((young_modulus, nu), ))
    material.Plastic(table=((yield_strength, 0.0), )) # elasto-plastic model is used here
          
    ## Selecting sets      
    # Select the entire unit cell tesselation  
    bounds = (((-np.sqrt(3)/4)*l,-1.5*l),((0.75*np.sqrt(3)*l),1.5*l)) # ((x_min,y_min),(x_max,y_max))
    unit_cell_set = part.Set(faces=part.faces.getByBoundingBox(-3*l,-3*l,0,7*l,7*l,0), name='UNIT_CELL_TESSELATION')
    
    # Section creation and assignment
    model.HomogeneousSolidSection(material=material_name, name='SECTION', thickness=1.0)
    part.SectionAssignment(region=unit_cell_set, sectionName='SECTION', thicknessAssignment=FROM_SECTION)
    
    # Create instance
    asmb = model.rootAssembly
    asmb.DatumCsysByDefault(CARTESIAN)
    instance = asmb.Instance(dependent=ON, name = 'INST', part=part)
    
    # Create buckling to determine buckling stress    
    model.BuckleStep(maxEigen=None, name='BuckleStep',numEigen=3, previous='Initial', vectors=10)
    
    # Create static step for stiffness/ strength analysis
    model.StaticStep(initialInc=0.01, maxInc=0.01, maxNumInc=1000, minInc=0.01, name='StaticStep', previous='BuckleStep')
    model.steps['StaticStep'].setValues(nlgeom=ON) # setting Nlgeom to ON
    
    
    
    # Output requests
    model.fieldOutputRequests.keys()
    model.fieldOutputRequests['F-Output-2'].setValues(variables=('S', 'U', 'EVOL','RF')) 
    
    # Generate the mesh on part
    elements_per_thickness = 5
    mesh_size = t_uni/(elements_per_thickness*(xi+(1-xi)/eta)) # this gives at least n elements per smallest thickness
    part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size)
    part.generateMesh()
    
    asmb.regenerate() # This is super important 
    
    # Create a reference point (outside the domain RP-1)  
    
    ref_point=asmb.ReferencePoint(point=(1.1*bounds[1][0], 1.1*bounds[1][1], 0))
    ref_id = ref_point.id
    ref_point= asmb.referencePoints[ref_id]
    
    ref_set = asmb.Set(name='REF', referencePoints=(ref_point, ))
    
    # Create sets for unit cell nodes with a reference point for each 
        # Create sets for unit cell nodes with a reference point for each 
    # We first generate the outter points in code and then use those to select edges

    def generate_edge_points(l):
        ep = [0]*6
        
        ep[0] = ((-np.sqrt(3)/4)*l,0.75*l)
        ep[1] = (0,1.5*l)
        ep[2] = ((0.75*np.sqrt(3))*l,0.75*l)
        ep[3] = (0.75*np.sqrt(3)*l,-0.75*l)
        ep[4] = (0,-1.5*l)
        ep[5] = ((-np.sqrt(3)/4)*l,-0.75*l)
            
        return ep

    
    e_points = generate_edge_points(l) # Generate the edge points
    num_edge_points = len(e_points)
    
    for idx in range(num_edge_points):
        p = e_points[idx]
        p_ref =asmb.ReferencePoint(point=(p+(0,)))
        p_ref_id = p_ref.id
        p_ref= asmb.referencePoints[p_ref_id]
    
        asmb.Set(name='n{}'.format(idx+2), referencePoints=(p_ref, ))
        asmb.regenerate()
        p_edges = instance.edges.getByBoundingSphere(p+(0,),1.1*t_uni)
        asmb.Surface(side1Edges=p_edges,name='surf-{}'.format(idx+2)) 
    
    
    # Coupling surfaces with edge points 
    for idx in range(num_edge_points):
        model.Coupling(controlPoint= asmb.sets['n{}'.format(idx+2)],
                                    couplingType=KINEMATIC,
                                    influenceRadius=WHOLE_SURFACE,
                                    localCsys=None,
                                    name='Coupling-{}'.format(idx+1),
                                    surface=asmb.surfaces['surf-{}'.format(idx+2)],
                                    u1=ON, u2=ON, ur3=ON)
    # Equation BC
    v_adjacent = [(6,3)] #(bottom,top)
    h_adjacent = [(4,2),(5,7)] # (right,left)
    
    
    # # # Equation BC
    # We start with the vertically adjacent nodes constraints
    eq_idx = 1
    for idx1,idx2 in v_adjacent:
        # Compatibility between nodes in DOF1
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 1), (-1.0, 'n{}'.format(idx2), 1)))
        eq_idx +=1 #increment
        # Compatibility between nodes and ref in DOF2
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 2), (-1.0, 'n{}'.format(idx2), 2), (1.0, 'REF', 2)))
        eq_idx +=1
        
    # Similarly for horizontally adjacent nodes
    for idx1,idx2 in h_adjacent:
        # Compatibility between nodes in DOF1
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 2), (-1.0, 'n{}'.format(idx2), 2)))
        eq_idx +=1 #increment
        # Compatibility between nodes and ref in DOF2
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 1), (-1.0, 'n{}'.format(idx2), 1), (1.0, 'REF', 1)))
        eq_idx +=1
    
    # # Finally we ensure compatibility in DOF6 (rotational) for all pairs
    for idx1,idx2 in v_adjacent+h_adjacent:
        # Compatibility between nodes in DOF6
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 6), (-1.0, 'n{}'.format(idx2), 6)))
        eq_idx +=1 #increment

        
    # Applying a load for the buckling step
    model.ConcentratedForce(cf2=-1.0,createStepName='BuckleStep',
                            distributionType=UNIFORM,
                            localCsys=None,
                            name='Load-1',
                            region=ref_set)
    
    # Applying displacements for the stress step 
    model.DisplacementBC(amplitude=UNSET, createStepName='StaticStep',
                          distributionType=UNIFORM,
                          name='BC-1',
                          region=ref_set,
                          u1=UNSET, 
                          u2=y_disp,
                          ur3=UNSET)
            
    modelJob = mdb.Job(model=model_name, name=job_name)      
    modelJob.submit(consistencyChecking=ON)
    modelJob.waitForCompletion()
    
def post_process(job_name, param_vector):
    
    # odb opening    
    odb_name = '{}.odb'.format(job_name)
    
    try:
        odb =  session.odbs[odb_name]
    except:    
        odb = session.openOdb(name=odb_name)
            
    # initialize an empty dictionary for sim results and lists for frame data
    sim_results = {}
    s11_list = []
    s22_list = []
    e11_list = []
    e22_list = []
    
    #zeroing the values so that they are in the outside scope
    e11,e22,s22,thick_buckle =  0,0,0,0

    ## Buckling analysis
    step = odb.steps['BuckleStep']
    frames = step.frames
    try:
        eigenv_text = frames[1].description
        eigenv = float(eigenv_text.split()[-1])
        thick_buckle =eigenv/(2*H)
    except:
        thick_buckle = None
        print('[WARNING] Could not find the first buckling load')
    
    
    
    
    
    ## Stress analysis (last step)
    # field outputs
    step = odb.steps['StaticStep'] # last step
    frames = step.frames # all frames
    frame = frames[-1]
    
    # Extract the data for reference node
    RP = odb.rootAssembly.nodeSets['REF'] # reference point set      
    for f in frames:
        ## Stresses
        force = f.fieldOutputs['RF'].getSubset(region=RP).values[0].data[1]
        # Stress is obtained by dividing the force by the effective area of application
        s22 = force/(2*H)       

        ## Strains
        U = f.fieldOutputs['U'].getSubset(region=RP)
        
        u1 = U.values[0].data[0]
        u2 = U.values[0].data[1]
        
        e22 = u2/(2*h)
        e11 = -u1/(2*H) 
              
        s22_list.append(s22)
        e11_list.append(e11)
        e22_list.append(e22)
    
    ## Relative density calculations
    # Sum element volumes to obtain the actual relative density
    volume = 0
    volumes = frame.fieldOutputs['EVOL'].values
    
    for v in volumes:
        volume+=v.data
        
    actual_rel_density = volume/(4*H*h) # from summing element volumes and dividing by number of full cells   
    uniform_rel_density = (2.0/np.sqrt(3))*(t_uni/l)*(1.0-(t_uni/l)/(2.0*np.sqrt(3))) # from geometric formula 
           
    
    t_uni_act = -np.sqrt(3)*(np.sqrt(1-actual_rel_density)-1)
    
    uniform_buckle = young_modulus*0.22*(t_uni_act/l)**3.0 # buckling load for uniform thickness hexagon
    # Converting lists to arrays
    e11_array = np.array(e11_list)
    e22_array =  np.array(e22_list)
    s22_array = np.array(s22_list)
    
    # # Calculating stiffness from the linear portion of the stress strain curve
    linear = 0.2/100
    linear_idx = np.argmax(abs(e22_array)>=abs(linear))
    
    thick_stiff = s22_array[linear_idx]/e22_array[linear_idx]
    uniform_stiff = (1.5*(actual_rel_density)**3.0)*young_modulus
    
    # Stress at plastic collapse
    thick_collapse = np.max(abs(s22_array))
    uniform_collapse = yield_strength*(2.0/3.0)*(t_uni_act/l)**2.0
    
    
        ## Stiffnesses
     
    sim_results['stiffness_ratio'] = thick_stiff/uniform_stiff
    #sim_results['stress_ratio'] = min(thick_buckle,thick_collapse)/min(uniform_buckle, uniform_collapse)
    #sim_results['actual_del_density'] = actual_rel_density
    
    odb.close()
    
    '''Pickling the data'''
    pickle_name = job_name +'_results.pkl'    
    f = open(pickle_name,'wb')
    pickle.dump(sim_results,f)
    f.close()    
    
    return sim_results
    


'''This section is used for testing inside ABAQUS'''
if __name__ == '__main__':
    # random naming is added to avoid the problem with lock files
    random_idx = random.randint(0,1e6)
    model_name = 'hex_solid_thick_buckle'
    job_name = model_name + '_test_' + str(random_idx)
    parametrization = {'eta':0.54, 'xi':0.36} # eta, xi
    create_sim(model_name,job_name,parametrization,save_cae=False)
    sim_results = post_process(job_name,parametrization)
    print(sim_results)