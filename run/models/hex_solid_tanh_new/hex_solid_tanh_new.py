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
from operator import attrgetter

from odbAccess import openOdb

import pickle
import os
import random 
import numpy as np
from win32com.client import Dispatch
import subprocess # for calling the solve


# UPDATED 
l = 1.0 # beam length (excluding the nodal regions)
h = 1.5*l # characteristic cell size in y direction (vertical span)
H = 0.5*np.sqrt(3)*l # characteristic cell size in x direction (horizontal span)

relative_density = 0.23 # imposed relative density

t_uni = -np.sqrt(3)*(np.sqrt(1-relative_density)-1) # uniform thickness corresponding to the given relative density (uniform)

## Material properties
#young_modulus = 893.15 #MPa
#yield_strength = 39.0

young_modulus = 7000.0
yield_strength = 130.0


nu = 0.3 # Poisson's ratio 

remote_strain = -0.3 # Set to more than 1% to capture plastic effects
y_disp = 2*h*remote_strain # this should be added in
#y_disp = -1.5
    
def beam_area_quart(parametrization, l_e, t_m):
    eta = parametrization['eta']
    xi = parametrization['xi']
    gamma = parametrization['gamma']

    gamma = float(gamma)

    # Calculate the intermediate values - include some edge cases
    if eta > 0.99:
        y_shift = t_m/2
        scale = 0.0        
    else:
        y_shift = t_m/4 * (1/eta + 1)
        scale = t_m/4 * (1/eta - 1)  
    
    if xi > 0.99:
        x_shift = 0.0
    elif xi < 0.01:
        x_shift = l_e/2
    else:
        x_shift = l_e * (1 - xi)/2
        

    
    # Indefinite integral
    def logcosh(x):
        # s always has real part >= 0
        s = np.sign(x) * x
        p = np.exp(-2 * s)
        return s + np.log1p(p) - np.log(2)    
    def I(x):
            
        if gamma < 0.01:
            tang = 0.0
            log_cosh = 0.0       
        else:        
            tang = np.tan(gamma)        
            
            if eta > 0.99:
                log_cosh = 0.0
            else:
                log_cosh = logcosh(tang/scale*(x_shift - x))
    
        return y_shift*x - scale**2*log_cosh/np.tan(gamma)

    return I(l_e/2) - I(0)
  

def rd_equivalence(t_m, parametrization, l, t, rel_dens):
    # Unpack the param values
    eta = parametrization['eta']
    xi = parametrization['xi']
    gamma = parametrization['gamma']
    # Calculate the effective beam length l_e
    l_e = l - t_m*np.sqrt(3)/(3*eta)  
    # Calculate the beam area
    A_beam_quart = beam_area_quart(parametrization, l_e, t_m)
    # Calculate nodal area
    A_node = np.sqrt(3)/4*(t_m/eta)**2
    # Calculate total area
    A = A_node + 6*A_beam_quart
    # Area in uniform thickness
    A_uni = 3*np.sqrt(3)*l**2/4

    return A/A_uni - rel_dens

def newton_raphson(f, x0, dfdx, tol=1e-12, maxiter=100):
    x = x0
    for i in range(maxiter):
        fval = f(x)
        if abs(fval) < tol:
            return x
        x -= fval / dfdx(x)
    raise RuntimeError("Failed to converge in Newton-Raphson")

def solve_parametrization(parametrization, l, t, rel_dens):
    x0 = t/2
    # Define the derivative of the function for use in the Newton-Raphson solver
    dfdx = lambda x: (rd_equivalence(x + 1e-6, parametrization, l, t, rel_dens) -
                      rd_equivalence(x - 1e-6, parametrization, l, t, rel_dens)) / (2e-6)
    # Use the Newton-Raphson method to find the solution
    t_m_sol = newton_raphson(lambda x: rd_equivalence(x, parametrization, l, t, rel_dens),
                             x0=x0, dfdx=dfdx)
    
    print("t_m/ t_uni ratio: " + str(t_m_sol/t_uni))
    # Recalculate the effective length
    eta = parametrization['eta']
    xi = parametrization['xi']
    gamma = parametrization['gamma']

    l_e_sol = l - t_m_sol*np.sqrt(3)/(3*eta)
    
    return t_m_sol, l_e_sol  

def thickened_analytical(parametrization):
    # Plastic collapse only
    eta = parametrization['eta']
    xi = parametrization['xi']
    
    analytical_stiffness_ratio = eta**3/((1-xi+eta*xi)**3*(xi**3+eta**3-eta**3*xi**3))
    if eta**2 > xi:
        analytical_stress_ratio = 1/(1-xi+xi*eta)**2
    else:
        analytical_stress_ratio = (1/xi)*(eta**2/(1-xi+xi*eta)**2)
        
    return analytical_stiffness_ratio, analytical_stress_ratio
    
    

def create_sim(model_name,job_name,parametrization,save_cae=False):
    '''Solving the parametrization'''
    t_m_sol, l_e_sol = solve_parametrization(parametrization, l, t_uni, relative_density)
    
    eta = parametrization['eta']
    xi = parametrization['xi']
    gamma = parametrization['gamma']
    
    '''CATIA PART'''
    # Connecting to windows COM
    CATIA = Dispatch('CATIA.Application')
    # optional CATIA visibility
    CATIA.Visible = True
    CATIA.DisplayFileAlerts = False
    
    # Check if automatic measure update is on and enable if necessary
    settingControllers = CATIA.SettingControllers
    measureSettingAtt = settingControllers.Item("CATSPAMeasureSettingCtrl")

    # Check if PartUpdateStatus is already True
    if not measureSettingAtt.PartUpdateStatus:
        # Enable PartUpdateStatus
        measureSettingAtt.PartUpdateStatus = True
        measureSettingAtt.Commit()
        print("PartUpdateStatus has been enabled.")
    
    partDocument = CATIA.ActiveDocument
    part = partDocument.Part
    
    parameters = part.Parameters    


    effective_length = parameters.Item('l_e')
    middle_thickness = parameters.Item('t_m')
    
    # Scaling is needed as CATIA operates in mm
    effective_length.Value = l_e_sol*1e3
    middle_thickness.Value = t_m_sol*1e3
    
    xi_param = parameters.Item('xi')
    eta_param = parameters.Item('eta')
    gamma_param = parameters.Item('gamma')
    
    try:
        xi_param.Value = xi
        eta_param.Value = eta
        gamma_param.Value = gamma
        part.Update()
        
    except Exception:
        print('The trial parametrization is incorrect')

    # Extract true measures
    t_min  = 2*parameters.Item('t_min/2\Length').Value/1e3
    global l_true
    l_true = parameters.Item('L\Length').Value/1e3   
    
    # THIS STILL NEEDS FIXING CAUSE IT IS NOT TRANSFERRABLE AS OF NOW
    catia_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()),'..','catia','%s.stp')%model_name)
    catia_dir = "D:/Temp/f_azam/IGOR/bayes-opt-for-abaqus/run/catia/hex_solid_tanh_new.stp"
    
    # This should go up one directory and then to catia folder to make sense
    partDocument.ExportData(catia_dir, "stp")
    
    '''ABAQUS PART'''  
    model = mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)
    
    # Delete 'Model-1' if it's already here
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']   
        
    # Import the step file    
    mdb.openStep(catia_dir, scaleFromFile=OFF)
    sketch_name = 'unit_cell_tanh'
    part_name = 'HEX_GT_BUCKLE'
    part = model.PartFromGeometryFile(combine=False,
                                      dimensionality=TWO_D_PLANAR,
                                      geometryFile=mdb.acis,
                                      name=part_name,
                                      scale=0.001,
                                      type=DEFORMABLE_BODY)
                                      
    part = model.parts['HEX_GT_BUCKLE']
    
    
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
    bounds = (((-np.sqrt(3)/4)*l_true,-1.5*l_true),((0.75*np.sqrt(3)*l_true),1.5*l_true)) # ((x_min,y_min),(x_max,y_max))
    unit_cell_set = part.Set(faces=part.faces.getByBoundingBox(-3*l_true,-3*l_true,0,7*l_true,7*l_true,0), name='UNIT_CELL_TESSELATION')
    
    # Section creation and assignment
    model.HomogeneousSolidSection(material=material_name, name='SECTION', thickness=1.0)
    part.SectionAssignment(region=unit_cell_set, sectionName='SECTION', thicknessAssignment=FROM_SECTION)
    
    # Create instance
    asmb = model.rootAssembly
    asmb.DatumCsysByDefault(CARTESIAN)
    instance = asmb.Instance(dependent=ON, name = 'INST', part=part)
    
    # Create buckling to determine buckling stress    
    model.BuckleStep(maxEigen=None, name='BuckleStep',numEigen=3, previous='Initial', vectors=10, maxIterations=500)
    
    # Create static step for stiffness/ strength analysis
    model.StaticStep(initialInc=0.01, maxInc=0.01, maxNumInc=1000, minInc=1e-15, name='StaticStep', previous='BuckleStep')
    model.steps['StaticStep'].setValues(nlgeom=ON) # setting Nlgeom to ON
    
    # Add stabilization
    # model.steps['StaticStep'].setValues(
    # adaptiveDampingRatio=0.05, continueDampingFactors=False, 
    # stabilizationMagnitude=0.0002, stabilizationMethod=
    # DISSIPATED_ENERGY_FRACTION)
    
    # Output requests
    model.fieldOutputRequests.keys()
    model.fieldOutputRequests['F-Output-2'].setValues(variables=('S', 'U', 'EVOL','RF'))

    # Perform auto-partition of the surface
    part.PartitionFaceByAuto(part.faces[0])
    
    # Generate the mesh on part                        
    elements_per_thickness = 20
    mesh_size = t_min/elements_per_thickness # this gives at least n elements per smallest thickness
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

    def generate_edge_points(l_true):
        ep = [0]*6
        
        ep[0] = ((-np.sqrt(3)/4)*l_true,0.75*l_true)
        ep[1] = (0,1.5*l_true)
        ep[2] = ((0.75*np.sqrt(3))*l_true,0.75*l_true)
        ep[3] = (0.75*np.sqrt(3)*l_true,-0.75*l_true)
        ep[4] = (0,-1.5*l_true)
        ep[5] = ((-np.sqrt(3)/4)*l_true,-0.75*l_true)
            
        return ep

    
    e_points = generate_edge_points(l_true) # Generate the edge points
    num_edge_points = len(e_points)
    
    for idx in range(num_edge_points):
        p = e_points[idx]
        p_ref =asmb.ReferencePoint(point=(p+(0,)))
        p_ref_id = p_ref.id
        p_ref= asmb.referencePoints[p_ref_id]
    
        asmb.Set(name='n{}'.format(idx+2), referencePoints=(p_ref, ))
        asmb.regenerate()

        # Get the edge that is closest to the edge point
        p_edge = instance.edges.findAt(coordinates=((p+(0,)),))
        
        if len(p_edge) == 0:
            p_edge = instance.edges.getByBoundingSphere(p+(0,),t_min)

        
        asmb.Surface(side1Edges=p_edge, name='surf-{}'.format(idx+2))        
    
    
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
    
    
    # # # # Equation BC    
           
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
    
def post_process(job_name, parametrization):
    # Redefine values based on l_true
    h_true = 1.5*l_true # characteristic cell size in y direction (vertical span)
    H_true = 0.5*np.sqrt(3)*l_true # characteristic cell size in x direction (horizontal span)
    
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
    force_list = []
    u22_list = []
    
    #initializing the values so that they are in the outside scope
    e11, e22, s22, thick_buckle =  0,0,0,0

    ## Buckling analysis
    step = odb.steps['BuckleStep']
    frames = step.frames
    try:
        eigenv_text = frames[1].description
        eigenv = float(eigenv_text.split()[-1])
        thick_buckle =eigenv/(2*H_true)
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
        s22 = force/(2*H_true)       

        ## Strains
        U = f.fieldOutputs['U'].getSubset(region=RP)
        
        u1 = U.values[0].data[0]
        u2 = U.values[0].data[1]
        
        e22 = u2/(2*h_true)
        e11 = -u1/(2*H_true) 
              
        s22_list.append(s22)
        e11_list.append(e11)
        e22_list.append(e22)
        force_list.append(force)
        u22_list.append(u2)
    
    ## Relative density calculations
    # Sum element volumes to obtain the actual relative density
    volume = 0
    volumes = frame.fieldOutputs['EVOL'].values
    
    for v in volumes:
        volume+=v.data
        
    actual_rel_density = volume/(4*h_true*H_true) # from summing element volumes and dividing by number of full cells   
    uniform_rel_density = (2.0/np.sqrt(3))*(t_uni/l)*(1.0-(t_uni/l)/(2.0*np.sqrt(3))) # from geometric formula (perfect)

    # For the comparisons to work, we find the actual corresponding uniform thickness
    t_uni_act = -np.sqrt(3)*(np.sqrt(1-actual_rel_density)-1)
    # Converting lists to arrays
    e11_array = np.array(e11_list)
    e22_array =  np.array(e22_list)
    s22_array = np.array(s22_list)
    
    
    force_array = np.array(force_list)
    u22_array = np.array(u22_list)
   
    # Find the linear portion of the curve
    linear = 0.2/100  # Define a threshold for the linear portion of the curve
    linear_idx = np.argmax(abs(e22_array)>=abs(linear))  # Find the index of the first element that exceeds the threshold    
    # Linear fit requires at least 2 values, therefore if linear falls below that, we will just take first few entries
    linear_idx = max(linear_idx, 3)
    
    s22_linear = s22_array[:linear_idx]  # Extract the linear portion of the s22 array
    e22_linear = e22_array[:linear_idx]  # Extract the linear portion of the e22 array
    
    # Calculate the Young's modulus using a polynomial fit    
    linear_fit_coefficients = np.polyfit(e22_linear, s22_linear, 1)  # Perform a least-squares polynomial fit and calculate the coefficients of the linear equation    
    thick_stiff = linear_fit_coefficients[0]
    
    

    # Stress at plastic collapse
    thick_collapse = np.max(abs(s22_array))
    
    
    # UNIFORM PROPS (ANALYTICAL)
    #uniform_stiff = (1.5*(actual_rel_density)**3.0)*young_modulus
    #uniform_collapse = yield_strength*(2.0/3.0)*(t_uni_act/l_true)**2.0
    #uniform_buckle = young_modulus*0.22*(t_uni_act/l_true)**3.0 # buckling load for uniform thickness hexagon  
    
    
    # UNIFORM PROPS (EXPERIMENTAL (MPa), RD 23)
    uniform_stiff = 139.024
    uniform_collapse = 4.06
    uniform_buckle = 154.16
    
    # UNIFORM PROPS (EXPERIMENTAL (MPa), RD 10)
    # uniform_stiff = 1.35
    # uniform_collapse = 0.222
    # uniform_buckle = 0.129
    
    ## Strength   
    #sim_results['thick_collapse'] = thick_collapse
    #sim_results['uniform collapse'] =  uniform_collapse
    
    #sim_results['uniform_buckle'] = uniform_buckle
    #sim_results['thick_buckle'] = thick_buckle
    
    # Stiffnessas 
    #sim_results['thick_stiff'] = thick_stiff
    #sim_results['uniform_stiff'] = uniform_stiff
         
    sim_results['stiffness_ratio'] = thick_stiff/uniform_stiff
    #sim_results['stress_ratio'] = thick_buckle/uniform_buckle
    sim_results['stress_ratio'] = min(thick_buckle, thick_collapse)/ min(uniform_stiff, uniform_buckle)
    
    # For monitoring the results
    # Calculate analytical stiffness and stress ratios
    #analytical_stiffness_ratio, analytical_stress_ratio = thickened_analytical(parametrization)
    
    #sim_results['stiffness_analytical_fea_error'] = 100*(analytical_stiffness_ratio - sim_results['stiffness_ratio'])/sim_results['stiffness_ratio']
    #sim_results['stress_analytical_fea_error'] = 100*(analytical_stress_ratio - sim_results['stress_ratio'])/sim_results['stress_ratio']
    
    sim_results['thick_buckle/thick_collapse'] = thick_buckle/thick_collapse
    sim_results['rel_density_error'] = 100*(relative_density - actual_rel_density)/actual_rel_density 
    
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
    model_name = 'hex_solid_tanh_new'
    job_name = model_name + '_test_' + str(random_idx)

    parametrization = {'eta': 0.1, 'xi':0.5576523731281741, 'gamma':0.22607099997088867} # eta, xi, gamma
    create_sim(model_name,job_name,parametrization,save_cae=False)
    sim_results = post_process(job_name,parametrization)
    print(sim_results)