import nbformat
import os

dir_path = '/Users/gyuminkang/Desktop/m5-forecasting-accuracy/visualizations'
nb1_path = os.path.join(dir_path, 'm5_analysis_and_results.ipynb')
nb2_path = os.path.join(dir_path, 'm5_comprehensive_visualization.ipynb')
out_path = os.path.join(dir_path, 'm5_final_report.ipynb')

if os.path.exists(nb1_path) and os.path.exists(nb2_path):
    nb1 = nbformat.read(nb1_path, as_version=4)
    nb2 = nbformat.read(nb2_path, as_version=4)
    
    # Merge cells: Visualizations first (as they are the architectural overview), then Analysis?
    # Or Analysis first, then Visualizations? 
    # Usually EDA (Analysis) -> Architecture/Results (Visualizations).
    # Let's put Analysis first, then Visualizations.
    
    # Create a markdown cell to act as a separator
    separator_cell = nbformat.v4.new_markdown_cell(source="# --- Part 2: Comprehensive Visualizations & Architecture ---")
    
    nb1.cells.extend([separator_cell])
    nb1.cells.extend(nb2.cells)
    
    nbformat.write(nb1, out_path)
    print(f"Successfully merged into {out_path}")
else:
    print("One of the notebooks does not exist.")
