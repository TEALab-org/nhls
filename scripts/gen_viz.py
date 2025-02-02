import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def parse_dot(dot_file):
    with open(dot_file, 'r') as f:
        content = f.read()

    # Extracting all the nodes and their attributes
    nodes = {}
    for match in re.finditer(r'(\S+) \[label="([^"]+)"\]', content):
        node_id = match.group(1)
        label = match.group(2)
        
        # Extract relevant data from the label
        method_match = re.search(r"^\S+:\s*(\S+)", label)
        steps_match = re.search(r"steps: (\d+)", label)
        in_match = re.search(r"in: \[\[(\d+)\], \[(\d+)\]\]", label)
        out_match = re.search(r"out: \[\[(\d+)\], \[(\d+)\]\]", label)

        if method_match and steps_match and in_match and out_match:
            method = method_match.group(1)
            steps = int(steps_match.group(1))
            
            # Extract the coordinate ranges
            in_coord = (int(in_match.group(1)), int(in_match.group(2)))
            out_coord = (int(out_match.group(1)), int(out_match.group(2)))
            
            nodes[node_id] = {
                'method': method,
                'steps': steps,
                'in_coord': in_coord,  # Range as a tuple
                'out_coord': out_coord  # Range as a tuple
            }
    
    return nodes


def visualize_grid(nodes, grid_size=(251, 1000), output_file="output.png"):
    # Create a blank grid (white background)
    grid = np.zeros(grid_size)
    
    # Define a numerical representation for each method
    method_map = {'DIRECT': 1, 'PERIODIC': 2}
    
    # Define a colormap using ListedColormap
    colormap = colors.ListedColormap(['white', 'red', 'green'])
    
    # Fill the grid with appropriate colors based on method
    for node in nodes.values():
        steps_y = 250-node['steps']
        # Get the numerical representation for the method of this node
        method_value = method_map.get(node['method'])
        for i in range(node['out_coord'][0], node['out_coord'][1]+1):
            grid[steps_y][i] = method_value
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(grid, cmap=colormap, interpolation='nearest')
    
    # Add method labels to the regions
    for node_id, node in nodes.items():
        x, steps_y = node['out_coord'][0]+(node['out_coord'][1]-node['out_coord'][0])//2, 250-nodgit ['steps']

        #ax.text(x, steps_y, node_id, color='black', ha='center', va='center', fontsize=3, fontweight='bold')
    
    # Save the image
    plt.axis('off')
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    plt.show()

# Example usage
dot_file = "/Users/mikeybudney/Desktop/NHLS/nhls/target/heat_1d_ap_fft/plan.dot"  # Replace with the path to your DOT file
nodes = parse_dot(dot_file)  # Parse the DOT file to get nodes
print(nodes)
visualize_grid(nodes)
