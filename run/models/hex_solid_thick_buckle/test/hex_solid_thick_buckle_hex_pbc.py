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
h = 1.5*l # characteristic cell size in y direction (half vertical span)

relative_density = 0.05 # imposed relative density
t_uni = -np.sqrt(3)*(np.sqrt(1-relative_density)-1) # middle section thickness for the given relative density(uniform)

young_modulus = 70000.0 #MPa
yield_strength = 130.0
nu = 0.3 # Poisson's ratio 


remote_strain = -0.01# Set to more than 1% to capture plastic effects
y_disp = remote_strain*6*l

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
    
    # in_work_obj = part.HybridBodies.Item("Geometrical Set.1").HybridShapes.Item("Final Surface")
    # part.InWorkObject = in_work_obj
    
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
    # hard code this for now cause it is problemaic
    catia_dir = r'D:\Temp\kuszczak_i\Studentship\bayes-opt-for-abaqus\run\catia\hex_solid_thick_bucle.stp'
    print(catia_dir)
    
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
    bounds = ((-3*np.sqrt(3)*l/4,0),(9*l*sqrt(3)/4,6*l)) # ((x_min,y_min),(x_max,y_max))
    unit_cell_set = part.Set(faces=part.faces.getByBoundingBox(-3*l,-3*l,0,7*l,7*l,0), name='UNIT_CELL_TESSELATION')
    
    # Section creation and assignment
    model.HomogeneousSolidSection(material=material_name, name='SECTION', thickness=1.0)
    part.SectionAssignment(region=unit_cell_set, sectionName='SECTION', thicknessAssignment=FROM_SECTION)
    
    # Create instance
    asmb = model.rootAssembly
    asmb.DatumCsysByDefault(CARTESIAN)
    instance = asmb.Instance(dependent=ON, name = 'INST', part=part)
    
    # Create buckling to determine buckling stress    
    model.BuckleStep(maxEigen=None, name='BuckleStep',numEigen=5, previous='Initial', vectors=10)
    
    # Create static step for stiffness/ strength analysis
    model.StaticStep(initialInc=0.01, maxInc=0.01, maxNumInc=1000, minInc=0.01, name='StaticStep', previous='BuckleStep')
    model.steps['StaticStep'].setValues(nlgeom=ON) # setting Nlgeom to ON
    
    
    
    # Output requests
    print(model.fieldOutputRequests.keys())
    model.fieldOutputRequests['F-Output-2'].setValues(variables=('S', 'U', 'EVOL','RF')) 
    
    # Generate mesh on part
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
    
    def generate_hex(a):    
        h = a*(np.sqrt(3)/2) # height
        
        return [(0,a),(h,a/2),(h,-a/2),(0,-a),(-h,-a/2),(-h,a/2)] # boundary nodes
    
    o_points = generate_hex(1.5*l) # Generate the outter hexagon
    
    
    for idx in range(6):
        p = o_points[idx]
        p_ref =asmb.ReferencePoint(point=(p+(0,)))
        p_ref_id = p_ref.id
        p_ref= asmb.referencePoints[p_ref_id]
    
        asmb.Set(name='n{}'.format(idx+1), referencePoints=(p_ref, ))
        asmb.regenerate()
        p_edges = instance.edges.getByBoundingSphere(p+(0,),1.1*t_uni)
        asmb.Surface(side1Edges=p_edges,name='surf-{}'.format(idx+1))  
    
    # Equation BC
    v_adjacent = [(2,5)] #(top node idx, bottom node idx)
    h_adjacent = [(3,7),(4,6)] # (right node idx, left node idx)
    
    
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
        # Compatibility between nodes in DOF1
        model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 6), (-1.0, 'n{}'.format(idx2), 6)))
        eq_idx +=1 #increment
        
    for idx in range(6):
        model.Coupling(controlPoint= asmb.sets['n{}'.format(idx+1)],
                                        couplingType=KINEMATIC,
                                        influenceRadius=WHOLE_SURFACE,
                                        localCsys=None,
                                        name='Coupling-{}'.format(idx+1),
                                        surface=asmb.surfaces['surf-{}'.format(idx+1)],
                                        u1=ON, u2=ON, ur3=ON)
    
    
    
    
    
    
    # Create sets for unit cell nodes with a reference point for each 
    # We first generate the outter points in code and then use those to select edges

    # def generate_edge_points(l):
        # ep = [0]*16
        # ep[2] = (0,0) 
        # ep[5] = (9*l*sqrt(3)/4,3*l/4)
        # ep[11] = (0,6*l)
        # ep[15] = (-3*np.sqrt(3)*l/4,3*l/4)
        
        
        # hor_vec = (np.sqrt(3)*l,0) # horizontal stride to generate edge points
        # ver_vec = (0, 1.5*l) #vertical stride
        
        # # Function for adding tuples 
        
        # def add_tuples(a,b):
            # a_list = list(a)
            # b_list = list(b)
            # c = []
            # for (it1,it2) in zip(a_list,b_list):
                # c.append(it1+it2)
            # return tuple(c)
            
        # ep[3] = add_tuples(ep[2],hor_vec)
        # ep[4] = add_tuples(ep[3],hor_vec)
        
        # ep[6] = add_tuples(ep[5],ver_vec)
        # ep[7] = add_tuples(ep[6],ver_vec)
        # ep[8] = add_tuples(ep[7],ver_vec)
        
        # ep[10] = add_tuples(ep[11],hor_vec)
        # ep[9] = add_tuples(ep[10],hor_vec)
        
        # ep[14] = add_tuples(ep[15],ver_vec)
        # ep[13] = add_tuples(ep[14],ver_vec)
        # ep[12] = add_tuples(ep[13],ver_vec)
        
        # edge_points = ep[2:]    
        # return edge_points

    
    # e_points = generate_edge_points(l) # Generate the edge points
    # num_edge_points = len(e_points)
    # for idx in range(num_edge_points):
        # p = e_points[idx]
        # p_ref =asmb.ReferencePoint(point=(p+(0,)))
        # p_ref_id = p_ref.id
        # p_ref= asmb.referencePoints[p_ref_id]
    
        # asmb.Set(name='n{}'.format(idx+2), referencePoints=(p_ref, ))
        # asmb.regenerate()
        # p_edges = instance.edges.getByBoundingSphere(p+(0,),1.1*t_uni)
        # asmb.Surface(side1Edges=p_edges,name='surf-{}'.format(idx+2))    
    
    # # First we need to kinematically couple the tesselation edges to their allocated reference points
    # for idx in range(num_edge_points):
        # model.Coupling(controlPoint= asmb.sets['n{}'.format(idx+2)],
                                        # couplingType=KINEMATIC,
                                        # influenceRadius=WHOLE_SURFACE,
                                        # localCsys=None,
                                        # name='Coupling-{}'.format(idx+1),
                                        # surface=asmb.surfaces['surf-{}'.format(idx+2)],
                                        # u1=ON, u2=ON, ur3=ON)  
    
    # # For equation BCs it is useful to store the sets of vertically and horizontally adjacent nodes
    # v_adjacent = [(2,11),(3,10),(4,9)] #(top node idx, bottom node idx)
    # h_adjacent = [(8,12),(7,13),(6,14),(5,15)] # (right node idx, left node idx)
    
    
    # # # Equation BC
    # # We start with the vertically adjacent nodes constraints
    # eq_idx = 1
    # for idx1,idx2 in v_adjacent:
        # # Compatibility between nodes in DOF1
        # model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 1), (-1.0, 'n{}'.format(idx2), 1)))
        # eq_idx +=1 #increment
        # # Compatibility between nodes and ref in DOF2
        # model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 2), (-1.0, 'n{}'.format(idx2), 2), (1.0, 'REF', 2)))
        # eq_idx +=1
        
    # # Similarly for horizontally adjacent nodes
    # for idx1,idx2 in h_adjacent:
        # # Compatibility between nodes in DOF1
        # model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 2), (-1.0, 'n{}'.format(idx2), 2)))
        # eq_idx +=1 #increment
        # # Compatibility between nodes and ref in DOF2
        # model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 1), (-1.0, 'n{}'.format(idx2), 1), (1.0, 'REF', 1)))
        # eq_idx +=1
    
    # # Finally we ensure compatibility in DOF6 (rotational) for all pairs
    # for idx1,idx2 in v_adjacent+h_adjacent:
        # # Compatibility between nodes in DOF1
        # model.Equation(name='Equation-{}'.format(eq_idx), terms=((1.0, 'n{}'.format(idx1), 6), (-1.0, 'n{}'.format(idx2), 6)))
        # eq_idx +=1 #increment
    
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
    e11,e22,s22,s_critical =  0,0,0,0

    ## Buckling analysis
    step = odb.steps['BuckleStep']
    frames = step.frames
    try:
        eigenv_text = frames[1].description
        eigenv = float(eigenv_text.split()[-1])
        s_critical =eigenv/(1.5*sqrt(3)*l)
    except:
        s_critical = None
        print('[WARNING] Could not find the first buckling load')
  
    
    
    
    ## Stress analysis (last step)
    # field outputs
    step = odb.steps['StaticStep'] # last step
    frames = step.frames # all frames
    frame = frames[-1]
    print(frame.fieldOutputs)
    
    # Extract the data for reference node
    RP = odb.rootAssembly.nodeSets['REF'] # reference point set      
    for f in frames:
        ## Stresses
        force = f.fieldOutputs['RF'].getSubset(region=RP).values[0].data[1]
        # Stress is obtained by dividing the force by the effective area of application
        s22 = force/(1.5*sqrt(3)*l)
        

        ## Strains
        U = f.fieldOutputs['U'].getSubset(region=RP)
        
        u1 = U.values[0].data[0]
        u2 = U.values[0].data[1]
        
        e22 = u2/(3*l)# 4 times h is simply the height of the rectangle which constains the unit cell 
        e11 = -u1/(1.5*sqrt(3)*l) # and this is the corresponding width
        
        
        
        s22_list.append(s22)
        e11_list.append(e11)
        e22_list.append(e22)
    
    ## Relative density calculations
    # Sum element volumes to obtain the actual relative density
    volume = 0
    volumes = frame.fieldOutputs['EVOL'].values
    
    for v in volumes:
        volume+=v.data
        
    actual_rel_density = volume/((1.5*np.sqrt(3)*(np.sqrt(3)*l)**2)) # from summing element volumes and dividing by number of full cells   
    uniform_rel_density = (2.0/np.sqrt(3))*(t_uni/l)*(1.0-(t_uni/l)/(2.0*np.sqrt(3))) # from geometric formula 
        
    ## Stiffnesses
    uniform_stiff = (1.5*(uniform_rel_density)**3.0)*young_modulus    
    
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
    sim_results['u2'] = u2 
    sim_results['u1'] = u1
    sim_results['poisson\'s ratio'] = -e11/e22 # this should be around 1
    sim_results['uniform stiffness'] = uniform_stiff    
    sim_results['thick_stiff_offset'] = thick_stiff
    sim_results['uniform_rel_density'] = uniform_rel_density
    sim_results['actual_rel_density'] = actual_rel_density    
    sim_results['rd_error_per_cell'] = 100*(uniform_rel_density-actual_rel_density)/actual_rel_density    
    sim_results['stiffness_ratio'] = thick_stiff/uniform_stiff
    sim_results['strength_ratio'] = abs(thick_yield)/uniform_yield
    sim_results['critical_stress'] = s_critical
    
    odb.close()
    
    '''Pickling the data'''
    pickle_name = job_name +'_results.pkl'    
    f = open(pickle_name,'wb')
    pickle.dump(sim_results,f)
    f.close()    
    
    return sim_results
    


'''This section is used for testing inside ABAQUS'''
if __name__ == '__main__':

    model_name = 'hex_solid_thick_buckle'
    job_name = model_name + '_test'
    parametrization = {'eta':0.99, 'xi':0.99} # eta, xi
    create_sim(model_name,job_name,parametrization,save_cae=False)
    sim_results = post_process(job_name,parametrization)
    print(sim_results)