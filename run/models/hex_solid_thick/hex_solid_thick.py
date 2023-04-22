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
import numpy as np
from win32com.client import Dispatch

l = 1.0 
h = 1.5*l # characteristic cell size
t_uni = 0.06 # middle section thickness (uniform)

young_modulus = 70000.0 #MPa
yield_strength = 130.0
nu = 0.3 # Poisson's ratio 


remote_strain = -0.001 # Set to 1% to capture plastic effects
y_disp = remote_strain*2*h

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
        
    catia_dir = os.path.abspath(os.path.join(os.getcwd(),'%s.stp')%model_name)
    partDocument.ExportData(catia_dir, 'stp')
    
    '''ABAQUS PART'''  
    model = mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)
    

    
    # Delete 'Model-1' if it's already here
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']   
        
    # Import the sketch 
    
    mdb.openStep(catia_dir, scaleFromFile=OFF)
    
    sketch_name = 'unit_cell_hex_thickened'
    part = model.PartFromGeometryFile(combine=False,
                                      dimensionality=TWO_D_PLANAR,
                                      geometryFile=mdb.acis,
                                      name='HEX_JT_SOLID',
                                      scale=0.001,
                                      type=DEFORMABLE_BODY)
    
    
    
    # Define the material
    material_name = 'ALUMINIUM'
    material = model.Material(name=material_name)
    material.Elastic(table=((young_modulus, nu), ))
    material.Plastic(table=((yield_strength, 0.0), )) # elasto-plastic model is used here
        
    
        
    ## Selecting sets      
    # Select the entire unit cell
    bound = 2*l
    unit_cell_set = part.Set(faces=part.faces.getByBoundingBox(-bound, -bound,0,bound,bound,0), name='UNIT_CELL')
    
    # Section creation and assignment
    model.HomogeneousSolidSection(material=material_name, name='SECTION', thickness=1.0)
    part.SectionAssignment(region=unit_cell_set, sectionName='SECTION', thicknessAssignment=FROM_SECTION)
    
    
    # Create instance
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    instance = a.Instance(dependent=ON, name = 'INST', part=part)
    
    # Create step 
    model.StaticStep(initialInc=0.01, maxInc=0.01, maxNumInc=1000, minInc=0.01, name='Step-1', previous='Initial')
    model.steps['Step-1'].setValues(nlgeom=ON) # setting Nlgeom to ON
    
    # Output requests
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'EVOL','RF')) 
    
    # Generate mesh on part
    mesh_size = t_uni/(4*(xi+(1-xi)/eta)) # this gives at least n elements per smallest thickness
    part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=mesh_size)
    part.generateMesh()
    
    a.regenerate() # This is super important 
    
    # Create a reference point
    
    
    ref_point=a.ReferencePoint(point=(0, bound, 0))
    ref_id = ref_point.id
    ref_point= a.referencePoints[ref_id]
    
    ref_set = a.Set(name='REF', referencePoints=(ref_point, ))
    
    # Create sets for unit cell nodes with a reference point for each 
    
    def generate_hex(a):    
        h = a*(np.sqrt(3)/2) # height
        
        return [(0,a),(h,a/2),(h,-a/2),(0,-a),(-h,-a/2),(-h,a/2)] # boundary nodes
    
    o_points = generate_hex(1.5*l) # Generate the outter hexagon
    
    
    for idx in range(6):
        p = o_points[idx]
        p_ref =a.ReferencePoint(point=(p+(0,)))
        p_ref_id = p_ref.id
        p_ref= a.referencePoints[p_ref_id]
    
        a.Set(name='n{}'.format(idx+1), referencePoints=(p_ref, ))
        a.regenerate()
        p_edges = instance.edges.getByBoundingSphere(p+(0,),1.1*t_uni)
        a.Surface(side1Edges=p_edges,name='surf-{}'.format(idx+1))    
    
      
    # Equation BC
    
    model.Equation(name='Equation-1', terms=((1.0, 'n4', 1), (-1.0, 'n1', 1))) # Correct
    model.Equation(name='Equation-2', terms=((1.0, 'n5', 1), (-1.0, 'n2', 1), (1.0, 'REF', 1))) # Correct
    model.Equation(name='Equation-3', terms=((1.0, 'n6', 1), (-1.0, 'n3', 1), (1.0, 'REF', 1))) # Correct 
    model.Equation(name='Equation-4', terms=((1.0, 'n4', 2), (-1.0, 'n1', 2), (1.0, 'REF', 2))) # Correct
    model.Equation(name='Equation-5', terms=((1.0, 'n5', 2), (-1.0, 'n2', 2), (0.5, 'REF', 2))) # Correct
    model.Equation(name='Equation-6', terms=((1.0, 'n6', 2), (-1.0, 'n3', 2), (-0.5, 'REF', 2))) # Correct
    model.Equation(name='Equation-7', terms=((1.0, 'n4', 6), (-1.0, 'n1', 6))) # Correct
    model.Equation(name='Equation-8', terms=((1.0, 'n5', 6), (-1.0, 'n2', 6))) # Correct
    model.Equation(name='Equation-9', terms=((1.0, 'n6', 6), (-1.0, 'n3', 6))) # Correct
    
    for idx in range(6):
        model.Coupling(controlPoint= a.sets['n{}'.format(idx+1)],
                                        couplingType=KINEMATIC,
                                        influenceRadius=WHOLE_SURFACE,
                                        localCsys=None,
                                        name='Coupling-{}'.format(idx+1),
                                        surface=a.surfaces['surf-{}'.format(idx+1)],
                                        u1=ON, u2=ON, ur3=ON)
    
    
    model.DisplacementBC(amplitude=UNSET, createStepName='Step-1',
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
    e11,e22,s22 =  0,0,0
    # Extract the data for reference node
    RP = odb.rootAssembly.nodeSets['REF'] # reference point set    
    
    # field outputs
    step = odb.steps[odb.steps.keys()[-1]] # last step
    frames = step.frames # all frames
    frame = frames[-1]
        
    for f in frames:
        ## Stresses
        force = f.fieldOutputs['RF'].getSubset(region=RP).values[0].data[1]
        s22 = force/(1.5*sqrt(3)*l)
        
        ## Strains
        U = f.fieldOutputs['U'].getSubset(region=RP)
        
        u1 = U.values[0].data[0]
        u2 = U.values[0].data[1]
        
        e11 = u1/(sqrt(3)*h)
        e22 = u2/(2*h)
        
        s22_list.append(s22)
        e11_list.append(e11)
        e22_list.append(e22)
    
    ## Relative density calculations
    # Sum element volumes to obtain the actual relative density
    volume = 0
    volumes = frame.fieldOutputs['EVOL'].values
    
    for v in volumes:
        volume+=v.data
        
    actual_rel_density = volume/(1.5*np.sqrt(3)*(np.sqrt(3)*l)**2) # from summing element volumes    
    uniform_rel_density = (2.0/np.sqrt(3))*(t_uni/l)*(1.0-(t_uni/l)/(2.0*np.sqrt(3))) # from geometric formula 
        
    ## Stiffnesses
    uniform_stiff = (1.5*(uniform_rel_density)**3.0)*young_modulus    
    thick_stiff = s22/e22
    
    # Converting lists to arrays
    e11_array = np.array(e11_list)
    e22_array =  np.array(e22_list)
    s22_array = np.array(s22_list)
    
    # Calculating offset yield strength
    offset = -0.2/100
    offset_idx = np.argmax(abs(e22_array)>=abs(offset))    
    thick_stiff = s22_array[offset_idx]/e22_array[offset_idx]
    offset_line = thick_stiff*(e22_array-offset)    
    intersect_idx = np.argwhere(np.diff(np.sign(s22_array - offset_line))).flatten()[0]
    
    thick_yield = s22_array[intersect_idx] # offset yield strength    
    uniform_yield = 0.5*(uniform_rel_density**2.0)*yield_strength


    
    # writing results into dictionary    
    # sim_results['poisson\'s ratio'] = -e11/e22 # this should be around 1
    # sim_results['uniform_rel_density'] = uniform_rel_density
    # sim_results['actual_rel_density'] = actual_rel_density
    
    # sim_results['rd_error'] = 100*(uniform_rel_density-actual_rel_density)/actual_rel_density    
    sim_results['stiffness_ratio'] = thick_stiff/uniform_stiff
    sim_results['strength_ratio'] = abs(thick_yield)/uniform_yield
    
    odb.close()
    
    '''Pickling the data'''
    pickle_name = job_name +'_results.pkl'    
    f = open(pickle_name,'wb')
    pickle.dump(sim_results,f)
    f.close()    
    
    return sim_results
    


'''This section is used for testing inside ABAQUS'''
if __name__ == '__main__':

    model_name = 'hex_solid_thick'
    job_name = model_name + '_test'
    parametrization = {'eta':0.60, 'xi':0.36} # eta, xi
    create_sim(model_name,job_name,parametrization,save_cae=False)
    sim_results = post_process(job_name,parametrization)
    print(sim_results)