from win32com.client import Dispatch
import os

def create_drawing(parametrization,product_name,part_name,save_name):
    # Extract the individual parameter values
    eta = parametrization['eta']
    xi = parametrization['xi']
    #gamma = parametrization['gamma']
    
    # Connecting to windows COM
    CATIA = Dispatch('CATIA.Application')
    # Set CATIA visibility
    CATIA.Visible = True
    CATIA.DisplayFileAlerts = False
    
    # CATIA documents
    document = CATIA.Documents
    
    # Open the part for drawing
    part_doc = document.Item(part_name)
    part = part_doc.Part

    parameters = part.Parameters
    
    # Modify the parameters based on the parametrization
    
    xi_param = parameters.Item('xi')
    eta_param = parameters.Item('eta')
    #gamma_param = parameters.Item('gamma')

    try:
        xi_param.Value = xi
        eta_param.Value = eta
        #gamma_param.Value = gamma
        part.Update()
    except Exception:
        print('The trial parametrization is incorrect')
    
    # Add a drawing 
    drawing = document.Add("Drawing")
    
    sheet = drawing.Sheets.ActiveSheet
    # sheet.Standard = "catISO"
    sheet.PaperSize = 13
    # sheet.Orientation = catPaperLandscape
    
    views = sheet.Views
    view = views.Add("Front View") 
    
    # Declare the part to draw in the front view
    
    
    generative_behaviour = view.GenerativeBehavior
        
    generative_behaviour.Document = part_doc.GetItem(product_name)
    generative_behaviour.DefineFrontView(1,0,0,0,1,0)
    
    view.Scale = 0.25
    
    view.x = 600
    view.y = 450
    
    
    
    # Update the view
    generative_behaviour.Update()
    sheet.Update()
    
    drawing = CATIA.ActiveDocument
     
    drawing.ExportData(save_name, "tif")
    drawing.Close() # closing should make the part document active for the consequent iteration
    
    active_window = CATIA.ActiveWindow
    active_window.Close()    
    
def generate_images_from_ax_client(ax_client):
    
    part_name = "hex_solid_thick_buckle.CATPart"
    product_name = "hex_solid_thick"
    save_dir = "D:\Temp\kuszczak_i\Studentship\photos_for_animation"
    
    parametrization_dict = {trial.arm.name: trial.arm.parameters for  trial in ax_client.experiment.trials.values()}
    
    for trial_num, param in parametrization_dict.items():
        parametrization = param
        save_name = os.path.join(save_dir,f'{product_name}_trial_{trial_num}.tif')
        print(save_name)
        
        create_drawing(parametrization,product_name,part_name,save_name)
    
    
if __name__=='__main__':
    
    parametrization_dict = {'0':{'eta':0.5, 'xi':0.4, 'gamma':0.7},
                            '1':{'eta':0.3, 'xi':0.7, 'gamma':0.5}}

    part_name = "hex_solid_tanh_buckle.CATPart"
    product_name = "hex_solid_tanh"

    save_dir = "D:\Temp\kuszczak_i\Studentship\photos_for_animation"
    
    for trial_num, param in parametrization_dict.items():
        parametrization = param
        save_name = os.path.join(save_dir,f'{product_name}_trial_{trial_num}.tif')
        print(save_name)
        
        create_drawing(parametrization,product_name,part_name,save_name)
        
        # now crop to size
        
        
    
    


