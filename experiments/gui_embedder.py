from datetime import datetime
import hashlib
import io
import os
import random
import traceback
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import torch
import networkx as nx
from lxml import etree as ET
from experiments.logger import setup_logger
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from torchvision import transforms
from torch_geometric.data import Data, Dataset
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AndroidScreenEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(AndroidScreenEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):  # x: (B, 3, H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GUIEmbedder:
    def __init__(
        self,
        log_dir: str = "app/logs/",
        emulator_name: str = "Unknown",
        app_name: str = "Unknown",
    ):
        self.log_dir = log_dir

        self.logger = setup_logger(
            f"{log_dir}/gui_embedder.log",
            emulator_name=emulator_name,
            app_name=app_name,
        )

        # Load pretrained semantic encoder (MiniLM is compact and fast)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder.eval()

        self.action_types = {
            "click": 0,
            "long_click": 1,
            "edit_number": 2,
            "edit_text": 3,
            "scroll_up": 4,
            "scroll_down": 5,
            "scroll_left": 6,
            "scroll_right": 7,
            "rotate_landscape": 8,
            "rotate_portrait": 9,
            "volume_up": 10,
            "volume_down": 11,
            "back": 12,
            # "context_click": 13,
        }

        self.input_type_keys = {
            "0": "none",  # No input type
            "1": "text",
            "2": "number",
        }

        self.eltext_embedding_dim = 32
        self.eldesc_embedding_dim = 16
        self.vis_dim = 64
        self.action_dim = len(self.action_types)
        self.input_types = len(self.input_type_keys)
        self.action_space_dim = (
            1
            + self.action_dim
            + self.input_types
            + 3
            + 6
            + self.eltext_embedding_dim
            + self.eldesc_embedding_dim
            + self.vis_dim
        )

        self.graph = nx.MultiDiGraph()
        self.unique_activity_ids = set()  # Track unique activity IDs
        self.is_debug = True
        self.patch_size = 28  # Size of the patches to extract from screenshots

        self.visual_encoder = AndroidScreenEncoder(output_dim=self.vis_dim).to(device)
        self.img_to_tensor = transforms.Compose(
            [
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),  # Converts to (C, H, W), values [0,1]
            ]
        )
        self.system_vector_tensors = []
        self.system_possible_actions = []

    @torch.no_grad()
    def _get_text_embedding(self, text: str, dimension: int) -> torch.Tensor:
        if not text.strip():
            return torch.zeros(dimension, dtype=torch.float32)

        inputs = self.tokenizer(text.strip(), return_tensors="pt", truncation=True)
        outputs = self.text_encoder(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Pad or trim to exactly `dimension`
        current_dim = embedding.shape[1]
        if current_dim < dimension:
            pad = torch.zeros(
                (1, dimension - current_dim),
                dtype=embedding.dtype,
                device=embedding.device,
            )
            embedding = torch.cat((embedding, pad), dim=1)
        else:
            embedding = embedding[:, :dimension]

        return F.normalize(embedding.squeeze(0), dim=0)

    def _extract_text_embeddings(self, element):
        """Returns a consistent 48-dim embedding (32 from text, 16 from desc)."""
        # Reserve 48 dims and fill accordingly
        out = torch.zeros(
            self.eltext_embedding_dim + self.eldesc_embedding_dim, dtype=torch.float32
        )
        text = (
            element.get("text", "") or element.get("name", "")
            if element.tag != "EditText"
            else "edit_text"
        )
        desc = element.get("content-desc", "") or element.get("contentDescription", "")
        if text.strip():
            emb_text = self._get_text_embedding(text, self.eltext_embedding_dim)
            out[: self.eltext_embedding_dim] = emb_text
        if desc.strip():
            emb_desc = self._get_text_embedding(desc, self.eldesc_embedding_dim)
            out[
                self.eltext_embedding_dim : self.eltext_embedding_dim
                + self.eldesc_embedding_dim
            ] = emb_desc
        return out, [text, desc]

    def hash_(self, text: str) -> float:
        h = hashlib.md5(text.encode()).hexdigest()
        return h

    def _input_type_vector(self, element) :
        """Extract input type information for EditText elements."""
        element_type = element.get("class", "") or element.get("className", "")
        if "EditText" in element_type:
            input_type_keys = list(self.input_type_keys.keys())
            input_type_value = element.get("inputType", "x")
            return [int(input_type_value == k) for k in input_type_keys]
        return [0] * len(self.input_type_keys)  # No input type [1,0,0] [0, 0, 0] [0,1,0]

    def _extract_position(
        self, bounds_str: str, screen_width: int, screen_height: int
    ) -> tuple:
        """Extract absolute and normalized position coordinates from bounds string."""
        if bounds_str and bounds_str != "unknown":
            try:
                bounds = (
                    bounds_str.replace("][", ",")
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                )
                if len(bounds) == 4:
                    left, top, right, bottom = map(int, bounds)
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2

                    # Absolute coordinates
                    abs_coords = [left, top, right, bottom, center_x, center_y]
                    # Normalized
                    norm_coords = [
                        left / screen_width,
                        top / screen_height,
                        right / screen_width,
                        bottom / screen_height,
                        center_x / screen_width,
                        center_y / screen_height,
                    ]
                    return abs_coords, norm_coords
            except Exception as e:
                self.logger.error(f"Failed to parse bounds '{bounds_str}': {e}")

        return [0] * 6, [0.0] * 6

    def _create_action_vector(self, input_type_vector: list, element):
        """Create action vector from multiple action event and input type vectors."""
        action_vector = [0] * len(self.action_types)
        # Element event handling capabilities (5 values)
        action_event_vector = [
            int(element.get("clickable", "false").lower() == "true"),
            int(
                element.get("long-clickable", "false").lower() == "true"
                or element.get("longClickable", "false").lower() == "true"
            ),
            int(element.get("scrollable", "false").lower() == "true"),
            int(element.get("checkable", "false").lower() == "true"),
            # int(element.get("context-clickable", "false").lower() == "true"),
        ]

        scroll_direction_vector = [
            int("ScrollView" in (element.get("class", "") or element.get("className", ""))),
            int("HorizontalScrollView" in (element.get("class", "") or element.get("className", ""))),
            int("RecyclerView" in (element.get("class", "") or element.get("className", ""))),
            int("ListView" in (element.get("class", "") or element.get("className", ""))),
        ]

        is_edit_text = "EditText" in (element.get("class", "") or element.get("className", ""))

        # Correlate action event vector with action types
        if action_event_vector[0]:  # clickable
            action_vector[self.action_types["click"]] = 1
        if action_event_vector[1]:  # long-clickable
            action_vector[self.action_types["long_click"]] = 1
        if action_event_vector[2]:  # scrollable
            if (
                scroll_direction_vector[0]
                or scroll_direction_vector[2]
                or scroll_direction_vector[3]
            ):  # Vertical scroll
                action_vector[self.action_types["scroll_down"]] = 1
                action_vector[self.action_types["scroll_up"]] = 1
            if scroll_direction_vector[1]:  # Horizontal scroll
                action_vector[self.action_types["scroll_left"]] = 1
                action_vector[self.action_types["scroll_right"]] = 1
        if action_event_vector[3]:  # checkable
            action_vector[self.action_types["click"]] = 1
        # if action_event_vector[4]:  # context-clickable
        #     action_vector[self.action_types["context_click"]] = 1
        #     pass
        if input_type_vector[1] or input_type_vector[2] or is_edit_text:  # text input
            action_vector[self.action_types["edit_text"]] = 1
            action_vector[self.action_types["edit_number"]] = 1
        
        return action_vector

    def action_vector_to_str(self, action_vector: list) -> str:
        """Convert action vector to a human-readable string."""
        actions = []
        for action, idx in self.action_types.items():
            if action_vector[idx] == 1:
                actions.append(action)
        return ", ".join(actions) if actions else "none"
    
    def _extract_ui_patches(self, screenshot: Image.Image, patch_size=28):
        W, H = screenshot.size
        w_steps = W // patch_size
        h_steps = H // patch_size
        arr = np.array(screenshot)
        graph = nx.Graph()
        # Map patch index to its average RGB
        patch_rgbs = {}
        idx = 0

        # Create nodes
        for i in range(h_steps):
            for j in range(w_steps):
                y0, x0 = i * patch_size, j * patch_size
                patch = arr[y0 : y0 + patch_size, x0 : x0 + patch_size, :]
                mean_rgb = tuple(patch.reshape(-1, 3).mean(axis=0).round().astype(int))
                patch_rgbs[idx] = ((j * patch_size, i * patch_size), mean_rgb)
                graph.add_node(idx)
                idx += 1

        # Connect neighbors if RGB-identical
        for node_id, ((x0, y0), rgb) in patch_rgbs.items():
            neighbors = [
                node_id + 1 if (node_id + 1) in patch_rgbs else None,
                node_id + w_steps if (node_id + w_steps) in patch_rgbs else None,
            ]
            for nb in neighbors:
                if nb is not None and patch_rgbs[nb][1] == rgb:
                    graph.add_edge(node_id, nb)

        # Extract representative patches
        reps = []
        for comp in nx.connected_components(graph):
            node = next(iter(comp))  # pick first
            reps.append(patch_rgbs[node][0])  # (x0, y0)

        return reps

    def match_patches_to_element(self, patch_tensors, element_bounds):
        """
        Matches all patches that are fully inside the given element bounds.

        Args:
            patch_tensors (list): List of tuples: ((x0, y0), tensor)
            element_bounds (list): [left, top, right, bottom, cx, cy]

        Returns:
            matched_patches:  list tensors that fit inside the element
        """
        left, top, right, bottom = element_bounds[:4]
        matched = []

        for (x0, y0), tensor in patch_tensors:
            x1 = x0 + self.patch_size
            y1 = y0 + self.patch_size

            if left <= x0 and x1 <= right and top <= y0 and y1 <= bottom:
                matched.append(tensor)

        return self.combine_patches_grid(matched)

    def combine_patches_grid(self, patch_list):
        """
        Combines patches into a grid image tensor (H_out, W_out, 3).

        Args:
            patch_list (List[Tensor]): List of (3, H, W) tensors

        Returns:
            Tensor: (3, H_out, W_out) tensor
        """
        if not patch_list:
            return torch.zeros((3, self.patch_size, self.patch_size))

        # Calculate grid size (square-ish layout)
        grid_w = int(np.ceil(np.sqrt(len(patch_list))))
        grid_h = int(np.ceil(len(patch_list) / grid_w))

        # Create empty grid tensor
        canvas = torch.zeros(
            3,
            grid_h * self.patch_size,
            grid_w * self.patch_size,
            dtype=patch_list[0].dtype,
        )

        for idx, patch in enumerate(patch_list):
            row = idx // grid_w
            col = idx % grid_w
            y0 = row * self.patch_size
            x0 = col * self.patch_size
            canvas[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] = patch

        return canvas

    # MAIN FUNCTION TO EMBED GUI
    def gui_embed(
        self,
        gui_hierarchy: str,
        current_activity: str = "unknown",
        screenshot: Image.Image = None,
        prev_action_vector: list = None,
        prev_selected_action_space_vector_tensor: torch.Tensor = None,
        prev_activity_id_hash: str = None,
        prev_elm_id_hash: str = "unknown",
    ):
        # DQT
        # GUI EMBEDDING: Embedding the GUI hierarchy into:
        # Action EMBEDDING:
        # Node vector of wigets [Bounds t-l-b-r-center, 5 ][1hot 7 - event executability (i.e., click-
        # able, long-clickable, checkable, and scrollable) and three for widget
        # status (i.e., enable, checked, and password), 7] [Content-Free Text Attributes-normalized 8 digit hash, 8]
        # [Content-Related Text Attributes, text and content-desc
        # are compressed to 32 and 16 dimensions via paraphrase-multilingual-MiniLM-L12-v2 trained
        # by Sentence Transformers ] => len = 77
        # 9 Actions: [1-hot] click, long-click, edit via number, edit via text, scroll in four different directions, and back.
        # Widget vector (node) length = 77
        # Action vector length = Widget vector length  + 9 Actions (1 hot) 2nd time = 86
        # State EMBEDDING:
        # Graph widget vector (node) and action lead to
        # GNN trainning

        # ADD
        # image patches from screenshot
        # build UI-connected graph
        try:
            # Parse the GUI XML
            root = ET.fromstring(gui_hierarchy.encode("utf-8"))
            # Extract screen properties
            screen_width = int(root.get("width", "720"))
            screen_height = int(root.get("height", "1080"))
            root_type = root.get("class", "") or root.get("className", "")
            root_id = root.get("resource-id", "") or root.get("resourceId", "")
            activity_id_hash = self.hash_(root_id + root_type + current_activity)
            self.unique_activity_ids.add(activity_id_hash)
            if prev_activity_id_hash is not None and not self.graph.has_edge(
                prev_activity_id_hash, activity_id_hash
            ):
                self.graph.add_edge(
                    prev_activity_id_hash,
                    activity_id_hash,
                    type="previous_state_of_activity",
                    action=[-1] * len(self.action_types),  # -1
                    action_space=torch.tensor(
                        [-1] * self.action_space_dim, dtype=torch.float32
                    ),
                )

            self.logger.debug(
                f"Screen: {current_activity}, Size: {screen_width}x{screen_height}, Hash: {activity_id_hash}"
            )

            interactive_elements = root.xpath(  # Select unique  # Multiple actions can be found in the same element
                ".//*["
                "@clickable='true' or @long-clickable='true' or @longClickable='true'"
                " or @scrollable='true' or @checkable='true'"
                " or @context-clickable='true'"  # stylus, right click
                " or @focusable='true'"
                " or self::android.widget.EditText"
                "]"
            )
            # Log the number of interactive elements found
            self.logger.debug(
                f"Found {len(interactive_elements)} unique interactive element"
            )

            if prev_action_vector is None:
                prev_action_vector = [0.0] * len(self.action_types)

            if prev_selected_action_space_vector_tensor is None:
                prev_selected_action_space_vector_tensor = torch.zeros(
                    self.action_space_dim, dtype=torch.float32
                )

            # Process each interactive element
            element_vector_tensors = []
            possible_actions = []
            patch_tensors = []
            vis_embeddings_tensor = None

            if screenshot:
                screenshot = Image.open(io.BytesIO(screenshot)).convert("RGB")
                reps = self._extract_ui_patches(screenshot, self.patch_size)
                # self.save_patch_reps(reps, file_path=f"{self.log_dir}/patch_reps.txt")
                # For each representative patch, crop and get embedding:
                os.makedirs(f"{self.log_dir}/patches", exist_ok=True)
                self.visualize_patch_grid_with_index(
                    screenshot,
                    reps,
                    self.patch_size,
                    save_path=f"{self.log_dir}/patches/patch_grid_{str(activity_id_hash).replace('.', '')}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.png",
                )

                for x0, y0 in reps:
                    patch_img = screenshot.crop(
                        (x0, y0, x0 + self.patch_size, y0 + self.patch_size)
                    )
                    # Save patch for debugging
                    # patch_img.save(
                    #     f"{self.log_dir}/patches/patch_{x0}_{y0}_{str(activity_id_hash).replace('.', '')}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.png"
                    # )
                    patch_tensor = self.img_to_tensor(patch_img)
                    patch_tensors.append(((x0, y0), patch_tensor))
            else:
                self.logger.warning("No screenshot provided, using empty patches.")
                patch_tensors = [((0, 0), torch.zeros((3, self.patch_size, self.patch_size)))]

            for i, element in enumerate(interactive_elements):
                # Element identification
                element_type = element.get("class", "") or element.get("className", "")
                element_id = element.get("resource-id", "") or element.get(
                    "resourceId", ""
                )
                element_bounds = element.get("bounds", "")
                element_identifier_hash = self.hash_(element_id + element_type)
                # Allows to identify same element in different states

                position_vector, norm_position_vector = self._extract_position(
                    element_bounds, screen_width, screen_height
                )

                position_vector_tensor_norm = torch.tensor(
                    norm_position_vector, dtype=torch.float32
                )
                # Element status flags (3 values)
                status_vector = [
                    int(element.get("enabled", "false").lower() == "true"),
                    int(element.get("checked", "false").lower() == "true"),
                    int(element.get("password", "false").lower() == "true"),
                ]

                input_type_vector = self._input_type_vector(element)
                text_vector_embedding, text_raw = self._extract_text_embeddings(element)
                action_vector = self._create_action_vector(
                    input_type_vector, element
                )

                input_type_raw = "none"
                try:
                    if 1 in input_type_vector:
                        input_type_raw = self.input_type_keys[str(input_type_vector.index(1))]
                except (ValueError, KeyError):
                    input_type_raw = "none"

                action_can_be_performed = {
                    "id_hash": element_identifier_hash,
                    "type": element_type,
                    "resource_id": element_id,
                    "position": position_vector,
                    "position_norm": norm_position_vector,
                    "screen_size": [screen_width, screen_height],
                    "actions": action_vector,  # 1hot, what action can be performed
                    "status": status_vector,  # 1hot
                    "text_raw": text_raw,
                    "input_type_raw": input_type_raw,  # str
                }
                possible_actions.append(action_can_be_performed)

                if self.is_debug:
                    element_info = {
                        "id_hash": element_identifier_hash,
                        "type": element_type,
                        "text": text_raw,
                        "position": position_vector,
                        "status": status_vector,  # 1hot
                        "action_vector": action_vector,  # 1hot, what action can be performed
                        "input_type": input_type_vector,  # 1hot
                        "text_embedding_size": len(text_vector_embedding),
                    }
                    # Dump element info to element.log file
                    with open(
                        f"{self.log_dir}/elements.log", "a", encoding="utf-8"
                    ) as element_log:
                        element_log.write(f"{element_info}\n")

                    # with open(f"{self.log_dir}/elements_interpretation.log", "a") as inf_log:
                    #     inf_log.write(
                    #         f"Element {i}:\n{self.interpret_element_info(element_info)}\n\n"
                    #     )

                patch_tensor = self.match_patches_to_element(
                    patch_tensors, position_vector
                )
                # self.logger.warning(
                #     f"Patch tensor for {element_type}-{text_raw}: {patch_tensor.shape}"
                # )
                vis_embeddings = torch.zeros((1, self.vis_dim), dtype=torch.float32)
                if patch_tensor is None:
                    vis_embeddings = torch.zeros((self.vis_dim), dtype=torch.float32)
                else:
                    vis_embeddings = self.visual_encoder(
                        patch_tensor.unsqueeze(0).to(device)
                    )
                    #self.logger.warning(f"Patch tensor shape: {vis_embeddings.shape}")
                    vis_embeddings = vis_embeddings.squeeze(0)
                    vis_embeddings = vis_embeddings.detach()

                # Create the complete element vector
                element_vector = torch.tensor(
                    [int(element_identifier_hash, 16) / float(2**192)]
                    + action_vector
                    + status_vector
                    + input_type_vector,
                    dtype=torch.float32,
                )

                element_vector_complete_tensor = torch.cat(
                    (
                        element_vector,
                        position_vector_tensor_norm,
                        text_vector_embedding,
                        vis_embeddings.cpu(),
                    ),
                    dim=0,
                )

                self.graph.add_node(
                    element_identifier_hash,
                    label=f"elm_id: {element_identifier_hash}",
                    type="element",
                    data=element_vector_complete_tensor,
                )
                self.graph.add_node(
                    activity_id_hash,
                    label=f"activity_id: {activity_id_hash}",
                    type="activity",
                )

                # Add edge from activity to element
                if not self.graph.has_edge(activity_id_hash, element_identifier_hash):
                    self.graph.add_edge(
                        activity_id_hash,
                        element_identifier_hash,
                        type="child_of_activity",
                        action=[-1] * len(self.action_types),  # -1
                        action_space=torch.tensor(
                            ([-1] * self.action_space_dim), dtype=torch.float32
                        ),  # -1
                    )
                if prev_elm_id_hash is not None and not self.graph.has_edge(
                    prev_elm_id_hash, element_identifier_hash
                ):
                    self.graph.add_edge(
                        prev_elm_id_hash,
                        element_identifier_hash,
                        type="previous_state_of_element",
                        action=prev_action_vector,
                        action_space=prev_selected_action_space_vector_tensor,
                    )

                # self.logger.debug(f"Element {i} vector length: {len(element_vector_complete)}")
                element_vector_tensors.append(element_vector_complete_tensor)
            if self.is_debug:
                with open(f"{self.log_dir}/elements.log", "a") as element_log:
                    element_log.write(
                        "\n\n================== EO XML =====================\n\n"
                    )
                # with open(f"{self.log_dir}/elements_interpretation.log", "a") as inf_log:
                #             inf_log.write(
                #                 "\n\n================== EO XML =====================\n\n"
                #             )
            ###########################
            # ADD SYSTEM ACTIONS
            ###########################
            self.get_system_action_space()
            element_vector_tensors.extend(
                self.system_vector_tensors
            )  # Add system actions to the action space
            possible_actions.extend(self.system_possible_actions)
            #############################
            # STACK TO ACTION SPACE VECTOR TENSOR
            #############################
            action_space_vector_tensor = (
                torch.stack(element_vector_tensors, dim=0)
                if element_vector_tensors
                else torch.zeros(
                    (0, self.state_dim), dtype=torch.float32, device=device
                ).unsqueeze(0)
            )
            self.logger.debug(
                f"Action space vector shape: {action_space_vector_tensor.shape}"
            )

            return (
                activity_id_hash,
                action_space_vector_tensor,
                possible_actions,
                vis_embeddings_tensor,
            )

        except Exception as e:
            self.logger.error(f"Error in GUI embedding: {str(e)}")
            # Return values consistent with normal return type
            self.logger.error(traceback.print_exc())
            return 0, torch.zeros(self.state_dim), [], torch.zeros(self.vis_dim)

    def get_system_action_space(self):
        if len(self.system_vector_tensors) != 0 or len(self.system_possible_actions) != 0:
            return  # system actions already generated

        back_action_space, back_action = self.gen_system_action("back")
        rotate_landscape_action_space, rotate_landscape_action = self.gen_system_action("rotate_landscape")
        rotate_portrait_action_space, rotate_portrait_action = self.gen_system_action("rotate_portrait")
        volume_up_action_space, volume_up_action = self.gen_system_action("volume_up")
        volume_down_action_space, volume_down_action = self.gen_system_action("volume_down")
        
        # Add back option to action space - prevent no action space
        self.system_vector_tensors.append(back_action_space)
        self.system_possible_actions.append(back_action)
        # Add rotate_landscape option to action space
        self.system_vector_tensors.append(rotate_landscape_action_space)
        self.system_possible_actions.append(rotate_landscape_action)
        # Add rotate_portrait option to action space
        self.system_vector_tensors.append(rotate_portrait_action_space)
        self.system_possible_actions.append(rotate_portrait_action)
        # Add volume_up option to action space
        self.system_vector_tensors.append(volume_up_action_space)
        self.system_possible_actions.append(volume_up_action)
        # Add volume_down option to action space
        self.system_vector_tensors.append(volume_down_action_space)
        self.system_possible_actions.append(volume_down_action)

        self.logger.debug(
            f"System action space generated: \n\n======\n\n{self.system_vector_tensors}"
        )
        self.logger.debug(
            f"System possible actions generated: \n\n======\n\n{self.system_possible_actions}"
        )

    def gen_system_action(self, action_name):
        """
        Generate a system action based on the given action name.
        """
        sys_act_str = "back_action"
        sys_act_type = "back"
        if action_name == "back":
            sys_act_str = "back_action"
            sys_act_type = "back"
        elif action_name == "rotate_landscape":
            sys_act_str = "rotate_landscape_action"
            sys_act_type = "rotate_landscape"
        elif action_name == "rotate_portrait":
            sys_act_str = "rotate_portrait_action"
            sys_act_type = "rotate_portrait"
        elif action_name == "volume_up":
            sys_act_str = "volume_up_action"
            sys_act_type = "volume_up"
        elif action_name == "volume_down":
            sys_act_str = "volume_down_action"
            sys_act_type = "volume_down"

        hash_ = str(self.hash_(sys_act_str))
        sys_action_vector = [0] * len(self.action_types)
        sys_action_vector[self.action_types[sys_act_type]] = 1
        sys_action_space_vector_tensor = torch.tensor(
            torch.cat(
                (
                    torch.tensor(
                        [int(hash_, 16) / float(2**192)]
                        + sys_action_vector
                        + [1, 0, 0]
                        + [0] * len(self.input_type_keys),  # No input type,
                        dtype=torch.float32,
                    ),
                    torch.tensor([0]*6,dtype=torch.float32),  # Position
                    self._get_text_embedding(sys_act_str, self.eltext_embedding_dim+ self.eldesc_embedding_dim),
                    torch.tensor([0]*(self.vis_dim),dtype=torch.float32),
                ), dim=0
            )
        )

        sys_action = (
            {
                "id_hash": hash_,
                "type": sys_act_type,
                "resource_id":  sys_act_str,
                "position": [0]*6,
                "position_norm": [0.0] * 6,
                "screen_size": [0,0],
                "actions": sys_action_vector,  # Only sys action
                "status": [1, 0, 0],  # Enabled, not checked, not password
                "input_type": [0] * len(self.input_type_keys),
            }
        )
        return sys_action_space_vector_tensor, sys_action

    def save_patch_reps(self, reps, file_path="patch_reps.txt"):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n======BG========\n")
            for x, y in reps:
                f.write(f"{x},{y}\n")
            f.write("\n=======EO=======\n")

    def graph_dataset_save(self):
        """
        Save the current graph to disk with improved layout and readability.
        """
        plt.figure(figsize=(12, 12))  # Adjust canvas size

        # Choose layout: 'dot' works well for DAGs and hierarchy
        try:
            pos = graphviz_layout(self.graph, prog="dot")
        except Exception:
            pos = nx.spring_layout(self.graph, k=0.15, iterations=50)

        # Draw with better spacing and reduced clutter
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=300,
            node_color="lightblue",
            edge_color="gray",
            font_size=8,
            arrows=True,
        )

        plt.savefig(f"{self.log_dir}/graph_test.png", dpi=300)
        plt.close()

    def visualize_patch_grid_with_index(
        self, screenshot, reps, patch_size=28, save_path="patch_grid_indexed.png"
    ):
        """
        Draws bounding boxes with patch indices on the screenshot for debugging.

        Args:
            screenshot (PIL.Image): The original screen image.
            reps (list of (x, y)): List of top-left coordinates for each patch.
            patch_size (int): Size of the square patch.
            save_path (str): Where to save the debug image.
        """
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        # Optional: load default font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        for idx, (x, y) in enumerate(reps):
            # Draw rectangle
            draw.rectangle(
                [x, y, x + patch_size, y + patch_size], outline="red", width=1
            )
            # Draw index
            draw.text((x + 2, y + 2), str(idx), fill="red", font=font)

        img.save(save_path)
