import nbformat
import re
from nbconvert import PythonExporter

def load_function_from_notebook(notebook_path, function_name):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(notebook)
    
    function_code = ""
    function_pattern = re.compile(rf"def {function_name}\(")

    cells = notebook['cells']
    for cell in cells:
        if cell.cell_type == 'code':
            cell_code = cell.source
            if function_pattern.search(cell_code):
                function_code += cell_code + "\n"
    
    if not function_code:
        raise ValueError(f"Function {function_name} not found in the notebook.")
    
    # Create a temporary module to store the executed function code
    module = {}
    exec(function_code, module)
    
    return module[function_name]

