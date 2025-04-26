import torch
from lxml import etree as ET
from experiments.logger import setup_logger
import subprocess
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F



class GUIEmbedder:
    def __init__(self, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown",
                 max_activities: int = 50, max_widgets: int = 500):
        
        self.logger = setup_logger(f"{log_dir}/gui_embedder.log", emulator_name=emulator_name, app_name=app_name)
        self.max_activities = max_activities
        self.max_widgets = max_widgets
        self.state_dim = max_activities + max_widgets + 4  # +4 for orientation, network, focused_text, scrollable
        self.activity_dict = {}
        self.widget_dict = {}
        self.logger.info(f"GUIEmbedder initialized with state dimension: {self.state_dim}")

        # Load pretrained semantic encoder (MiniLM is compact and fast)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.semantic_dim = 384  # MiniLM hidden size
        self.text_encoder.eval()

    @torch.no_grad() #test
    def get_text_embedding(self, text: str) -> torch.Tensor:
        if not text.strip():
            return torch.zeros(self.semantic_dim)
        inputs = self.tokenizer(text.strip(), return_tensors="pt", truncation=True)
        outputs = self.text_encoder(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return F.normalize(embedding.squeeze(0), dim=0)  # unit norm


    def get_device_orientation(self):
        try:
            orientation = subprocess.check_output(["adb", "shell", "dumpsys", "input"]).decode()
            return 1 if "orientation=1" in orientation else 0
        except Exception as e:
            self.logger.error(f"Failed to get orientation: {e}")
            return 0

    def get_network_status(self):
        try:
            network = subprocess.check_output(["adb", "shell", "settings", "get", "global", "airplane_mode_on"]).decode().strip()
            return 0 if network == "1" else 1
        except Exception as e:
            self.logger.error(f"Failed to get network status: {e}")
            return 1

    def get_activity_index(self, activity: str) -> int:
        if activity not in self.activity_dict:
            if len(self.activity_dict) < self.max_activities:
                self.activity_dict[activity] = len(self.activity_dict)
            else:
                return 0  # Default index if max reached
        return self.activity_dict[activity]

    def embed(self, gui_hierarchy: str) -> torch.Tensor:
        try:
            # Parse the GUI XML
            root = ET.fromstring(gui_hierarchy.encode('utf-8'))
            current_activity = root.get('activity', 'unknown')
            
            # Find interactive elements
            clickable_elements = (
                root.findall(".//*[@clickable='true']") +
                root.findall(".//*[@scrollable='true']") +
                root.findall(".//*[@class='android.widget.EditText']")
            )

            # Encode the activity (screen)
            activity_index = self.get_activity_index(current_activity)
            activity_vector = torch.zeros(self.max_activities)
            activity_vector[activity_index] = 1
            semantic_vectors = []

            # Encode widgets
            widget_vector = torch.zeros(self.max_widgets)
            for element in clickable_elements:
                widget_type = element.get('class', '')
                if 'EditText' in widget_type:
                    # Use resource-id for text inputs, ignore text
                    widget_id = element.get('resource-id', '')
                    widget_key = (widget_type, widget_id)
                else:
                    # Use text for other widgets
                    widget_text = element.get('text', '')
                    widget_key = (widget_type, widget_text)
                
                # Map the widget to an index
                widget_index = self.get_widget_index(widget_key)
                widget_vector[widget_index] = 1
                text_content = element.get('text', '') or element.get('content-desc', '')
                if text_content:
                    semantic_vectors.append(self.get_text_embedding(text_content))

            
            orientation = self.get_device_orientation()
            network = self.get_network_status()
            focused_text = 1 if any(element.get('focused') == 'true' and 'EditText' in element.get('class', '') for element in root.findall(".//*")) else 0
            scrollable = 1 if any(element.get('scrollable') == 'true' for element in root.findall(".//*")) else 0
            additional_features = torch.tensor([orientation, network, focused_text, scrollable], dtype=torch.float32)

            # Combine into full state vector (add other features as needed)
            #state_vector = torch.cat((activity_vector, widget_vector, additional_features))
            if semantic_vectors:
                semantic_vector = torch.mean(torch.stack(semantic_vectors), dim=0)
            else:
                semantic_vector = torch.zeros(self.semantic_dim)
            state_vector = torch.cat((activity_vector, #500
                                      widget_vector,  #50
                                      additional_features,  #4 
                                      semantic_vector)) #384 

            return state_vector

        except Exception as e:
            self.logger.error(f"Error in GUI embedding: {str(e)}")
            return torch.zeros(self.state_dim)

    def get_widget_index(self, widget_key: tuple) -> int:
        if widget_key not in self.widget_dict:
            if len(self.widget_dict) < self.max_widgets:
                self.widget_dict[widget_key] = len(self.widget_dict)
            else:
                return 0  # Default index if limit reached
        return self.widget_dict[widget_key]    