import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.data import Data
import networkx as nx
from experiments.logger import setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GINStateEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) implementation for state encoding
    Based on the "How Powerful are Graph Neural Networks?" paper
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        out_channels: int = 96,
        num_layers: int = 3,
        dropout: float = 0.5,
        train_eps: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        if num_layers > 1:
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

        # Final projection if needed
        self.final_projection = None
        if num_layers == 1:
            self.final_projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for GIN

        Args:
            data: PyTorch Geometric Data object containing x, edge_index, and optionally batch

        Returns:
            Graph-level embedding tensor
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        # If no batch is provided, create one (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply GIN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final projection
        if self.final_projection is not None:
            x = self.final_projection(x)

        # Global pooling to get graph-level representation
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "add":
            graph_embedding = global_add_pool(x, batch)
        else:
            # Fallback to mean pooling
            graph_embedding = global_mean_pool(x, batch)

        return graph_embedding


class GINEStateEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 64,
        out_channels: int = 96,
        num_layers: int = 3,
        dropout: float = 0.5,
        train_eps: bool = False,
        pooling: str = "mean",
        edge_attr_dim: int = 32,  # dimension of your concatenated edge features
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Build GINE layers
        # First layer: in_channels → hidden
        self.convs.append(
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                eps=0.0,
                train_eps=train_eps,
                edge_dim=edge_attr_dim,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    eps=0.0,
                    train_eps=train_eps,
                    edge_dim=edge_attr_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        if num_layers > 1:
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels),
                    ),
                    eps=0.0,
                    train_eps=train_eps,
                    edge_dim=edge_attr_dim,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

        # In case of single-layer config
        self.final_projection = None
        if num_layers == 1:
            self.final_projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(
            data, "batch", torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        )

        # Pass through GINEConv layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.final_projection is not None:
            x = self.final_projection(x)

        # Graph-level pooling
        if self.pooling == "mean":
            out = global_mean_pool(x, batch)
        else:
            out = global_add_pool(x, batch)
        return out


class GATv2StateEncoder(nn.Module):
    """
    Your existing GATv2 implementation (unchanged)
    """

    def __init__(self, in_channels=128, hidden_channels=64, out_channels=96, heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(
            hidden_channels * heads, hidden_channels // heads, heads=1
        )
        self.linear = nn.Linear(hidden_channels // heads, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        # If no batch is provided, create one (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.lin_in(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling
        graph_vector = global_mean_pool(x, batch)
        return self.linear(graph_vector)


class StateEmbedder:
    """
    Improved StateEmbedder with proper GIN integration
    """

    def __init__(
        self,
        log_dir: str = "app/logs",
        app_name: str = "unk",
        emulator_name: str = "unk",
        feature_dim: int = 128,
        out_dim: int = 96,
        hidden_channels: int = 64,
        num_layers: int = 3,
        action_space_dim: int = 151,
        action_dim: int = 13,
    ):
        self.log_dir = log_dir
        self.logger = setup_logger(
            f"{self.log_dir}/state_embedder.log",
            emulator_name=emulator_name,
            app_name=app_name,
        )
        self.logger.info("Initializing StateEmbedder with GIN and GATv2 support")

        self.feature_dim = feature_dim
        self.out_channels = out_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.gin_model = None
        self.gatv2_model = None
        self.gine_model = None  # For GINE model if needed
        self.action_space_dim = action_space_dim  
        self.action_dim = action_dim
        self.known_edge_types = [
                "child_of_activity",
                "previous_state_of_element",
                "previous_state_of_activity",
            ]
        self.edge_attr_dim = len(self.known_edge_types) + self.action_dim + self.action_space_dim # Adjust based on edge_attr vector

    def type_to_id(self, etype):
        type= [0]*len(self.known_edge_types)
        if etype in self.known_edge_types:
            type[self.known_edge_types.index(etype)] = 1
        return type

    def sample_1hop_subgraph(
        self, graph: nx.MultiDiGraph, activity_id_hash: str, max_nodes: int = 500
    ) -> nx.MultiDiGraph:
        """
        Your existing subgraph sampling function (unchanged)
        """
        nodes = {activity_id_hash}
        nodes.update(
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == "element" and graph.has_edge(activity_id_hash, n)
        )

        for neighbor in list(graph.predecessors(activity_id_hash)) + list(
            graph.successors(activity_id_hash)
        ):
            if len(nodes) >= max_nodes:
                break
            nodes.add(neighbor)
            nodes.update(
                n
                for n, d in graph.nodes(data=True)
                if d.get("type") == "element"
                and graph.has_edge(neighbor, n)
                and len(nodes) < max_nodes
            )

        for node in list(nodes):
            if len(nodes) >= max_nodes:
                break
            neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
            nodes.update(
                n
                for n in neighbors
                if graph.has_edge(node, n, type="previous_state_of_element")
                and len(nodes) < max_nodes
            )

        return graph.subgraph(nodes).copy()

    def convert_nx_to_pyg(
        self,
        graph: nx.MultiDiGraph,
        max_nodes: int = 512,
    ) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object
        """
        try:
            feature_dim = self.feature_dim
            # # Trim the graph if too large
            # if max_nodes is not None and graph.number_of_nodes() > max_nodes:
            #     graph = sample_1hop_subgraph(graph, activity_id_hash, max_nodes)

            # Sort and map nodes
            activity_nodes = [
                n for n, d in graph.nodes(data=True) if d.get("type") == "activity"
            ]
            element_nodes = [
                n for n, d in graph.nodes(data=True) if d.get("type") == "element"
            ]
            all_nodes = activity_nodes + element_nodes

            if not all_nodes:
                # Handle empty graph case
                return None

            node_map = {node: idx for idx, node in enumerate(all_nodes)}

            # Create [num node, feature_dim] graph features
            x = torch.zeros((len(all_nodes), feature_dim), dtype=torch.float32)
            for node_id, idx in node_map.items():
                node_data = graph.nodes[node_id].get("data", torch.zeros(feature_dim))
                if (
                    isinstance(node_data, torch.Tensor)
                    and node_data.shape[0] == feature_dim
                ):
                    x[idx] = node_data

            print(f"Node features shape: {x.shape}, Number of nodes: {len(all_nodes)}")

            # Create edge_index and edge_type
            edge_attrs = []
            edge_indexes = []

            for src, dst, data_dict in graph.edges(data=True):
                etype = data_dict.get("type")
                eaction_vector = data_dict.get("action", [0]*self.action_dim )
                eaction_space_vector_tensor = data_dict.get("action_space", torch.tensor([0]*self.action_space_dim,dtype=torch.float32))

                if src in node_map and dst in node_map:
                    edge_index = torch.tensor([node_map[src], node_map[dst]], dtype=torch.long)  # 2 hashes
                    edge_type = torch.tensor(self.type_to_id(etype), dtype=torch.float32)
                    eaction_vector = torch.tensor(eaction_vector, dtype=torch.float32)
                    edge_attr = torch.cat(
                        [edge_type, eaction_vector, eaction_space_vector_tensor]
                    )
                    if edge_attr.shape[0] != self.edge_attr_dim:
                        self.logger.warning(
                            f"Edge attribute dimension mismatch: expected {self.edge_attr_dim}, "
                            f"got {edge_attr.shape[0]}"
                        )
                        self.logger.warning(f" edge type: {edge_type.shape}, action vector: {eaction_vector.shape}, action space vector: {eaction_space_vector_tensor.shape}")
                        self.logger.warning(
                            f"Edge from {src} to {dst} with type {etype} has attributes: {edge_attr}"
                        )
                        edge_attr = F.pad(
                            edge_attr, (0, self.edge_attr_dim - edge_attr.shape[0]), "constant", 0
                        )
                    edge_attrs.append(edge_attr)
                    edge_indexes.append(edge_index)

            edge_indexes = torch.stack(edge_indexes, dim=0)
            edge_index_t = edge_indexes.t().contiguous()
            edge_attrs = torch.stack(edge_attrs, dim=0)

            return Data(
                x=x, edge_index=edge_index_t, edge_attr=edge_attrs
            )
        except Exception as e:
            self.logger.info(f"Error converting graph to PyG format: {e}")
            self.logger.error(traceback.print_exc())
            return None
        
    def _initialize_model(self, model_type: str) -> nn.Module:
        """Initialize the appropriate model"""
        if model_type == "gin":
            if self.gin_model is None:
                self.gin_model = GINStateEncoder(
                    in_channels=self.feature_dim,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=self.num_layers,
                    dropout=0.5,
                    train_eps=True,
                    pooling="mean",
                ).to(device)
                self.logger.info(
                    f"GIN model initialized with feature_dim={self.feature_dim}, "
                    f"hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, "
                    f"num_layers={self.num_layers}"
                )
            return self.gin_model

        elif model_type == "gatv2":
            if self.gatv2_model is None:
                self.gatv2_model = GATv2StateEncoder(
                    in_channels=self.feature_dim,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                ).to(device)
                self.logger.info(
                    f"GATv2 model initialized with feature_dim={self.feature_dim}, "
                    f"out_channels={self.out_channels}, hidden_channels={self.hidden_channels}"
                )
            return self.gatv2_model
        elif model_type == "gine":
            if not hasattr(self, "gine_model") or self.gine_model is None:
                self.gine_model = GINEStateEncoder(
                    in_channels=self.feature_dim,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=self.num_layers,
                    dropout=0.5,
                    train_eps=True,
                    pooling="mean",
                    edge_attr_dim=self.edge_attr_dim,  # adjust based on your edge_attr vector
                ).to(device)
                self.logger.info(
                    f"GINE model initialized with feature_dim={self.feature_dim}, "
                    f"hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, "
                    f"num_layers={self.num_layers}"
                )
            return self.gine_model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def extract_state_vector(
        self,
        graph: nx.MultiDiGraph,
        max_nodes: int = 500,
        model_type: str = "gine",
    ) -> torch.Tensor:
        """
        Extract state vector using specified model type

        Args:
            graph: NetworkX MultiDiGraph representing the app state
            activity_id_hash: Hash of the current activity
            action_space_vector_tensor: Tensor containing widget action vectors
            max_nodes: Maximum number of nodes to include in the graph
            model_type: Type of model to use ("gin" or "gatv2")

        Returns:
            State vector tensor with shape (1, out_channels)
        """
        try:
            # Initialize model
            model = self._initialize_model(model_type)

            # Convert graph to PyG format
            data = self.convert_nx_to_pyg(
                graph,
            )
            if data is None:
                self.logger.error("Failed to convert graph to PyG format")
                return torch.zeros(1, self.out_channels, dtype=torch.float32, device=device)
            data = data.to(device)
            # Ensure data has correct feature dimensions
            if data.x.size(1) != self.feature_dim:
                self.logger.warning(
                    f"Data feature dimension mismatch: expected {self.feature_dim}, "
                    f"got {data.x.size(1)}"
                )
                if data.x.size(1) < self.feature_dim:
                    data.x = F.pad(
                        data.x, (0, self.feature_dim - data.x.size(1)), "constant", 0
                    )
                else:
                    data.x = data.x[:, : self.feature_dim]


            # Forward pass
            model.eval()
            with torch.no_grad():
                state_vector = model(data)

            # Ensure output has batch dimension
            if state_vector.dim() == 1:
                state_vector = state_vector.unsqueeze(0)

            return state_vector

        except Exception as e:
            self.logger.error(f"Error in extract_state_vector: {e}")
            self.logger.error(traceback.format_exc())
            return torch.zeros(1, self.out_channels, dtype=torch.float32, device=device)

    def get_model_info(self, model_type: str = "gin") -> dict:
        """Get information about the current model"""
        model = self._initialize_model(model_type)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "model_type": model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "feature_dim": self.feature_dim,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "device": str(device),
        }
