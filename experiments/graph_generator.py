from typing import Any, Dict
import networkx as nx
from networkx.readwrite import graphml
from experiments.logger import setup_logger
import os

class GraphGenerator:
    def __init__(self, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.logger = setup_logger(f"{log_dir}/graph_generator.log", emulator_name=emulator_name, app_name=app_name)
        self.graph = nx.DiGraph()
        self.log_dir = log_dir

    def add_transition(self, state_id: str, action: Dict[str, Any], next_state_id: str):
        """Add a state transition to the graph."""
        self.graph.add_node(state_id, label=f"State {state_id}")
        self.graph.add_node(next_state_id, label=f"State {next_state_id}")
        self.graph.add_edge(state_id, next_state_id, action_type=action['type'], parameters=str(action.get('parameters', '')))
        self.logger.debug(f"Added transition: {state_id} -> {next_state_id} via {action['type']}")

    def save_graph(self, filename: str):
        """Save the graph in GraphML format."""
        try:
            output_path = os.path.join(self.log_dir, filename)
            graphml.write_graphml(self.graph, output_path)
            self.logger.info(f"Saved graph to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save graph: {str(e)}")

    def generate_html(self, output_file: str):
        """Generate an HTML visualization of the graph using Vis.js."""
        try:
            nodes = [{"id": n, "label": self.graph.nodes[n]["label"]} for n in self.graph.nodes]
            edges = [{"from": u, "to": v, "label": f"{self.graph.edges[u, v]['action_type']} {self.graph.edges[u, v]['parameters']}"}
                     for u, v in self.graph.edges]
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>State Transition Graph</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    #graph {{ width: 100%; height: 600px; border: 1px solid lightgray; }}
                </style>
            </head>
            <body>
            <div id="graph"></div>
            <script type="text/javascript">
                var nodes = {nodes};
                var edges = {edges};
                var container = document.getElementById('graph');
                var data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
                var options = {{ nodes: {{ shape: 'dot', size: 10 }}, edges: {{ arrows: 'to' }} }};
                var network = new vis.Network(container, data, options);
            </script>
            </body>
            </html>
            """
            output_path = os.path.join(self.log_dir, output_file)
            with open(output_path, 'w') as f:
                f.write(html_content)
            self.logger.info(f"Generated HTML visualization: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate HTML: {str(e)}")