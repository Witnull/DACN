import hashlib
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
    
    def get_widget_index(self, widget_key: tuple) -> int:
            if widget_key not in self.widget_dict:
                if len(self.widget_dict) < self.max_widgets:
                    self.widget_dict[widget_key] = len(self.widget_dict)
                else:
                    return 0  # Default index if limit reached
            return self.widget_dict[widget_key]

    def hash_md5(self, text: str = "aaaa") -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    # MAIN FUNCTION TO EMBED GUI
    def embed(self, gui_hierarchy: str, current_activity : str) -> torch.Tensor:
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

        try:
            activity_vector = torch.zeros(self.max_activities)
            widget_vector = torch.zeros(self.max_widgets)
            # Parse the GUI XML
            root = ET.fromstring(gui_hierarchy.encode('utf-8'))
            
            self.logger.debug(f"Current Activity: {current_activity}")
            
            # Find interactive elements
            interactive_elements = (
                root.findall(".//*[@clickable='true']") +
                root.findall(".//*[@long-clickable='true']") + root.findall(".//*[@longClickable='true']") +
                root.findall(".//*[@scrollable='true']") +
                root.findall(".//*[@checkable='true']") +  # Original 4 event executabile

                root.findall(".//*[@context-clickable='true']") +
                root.findall(".//*[@focusable='true']") +
                root.findall(".//*[@class='android.widget.EditText']")
            )

            # Log the number of interactive elements found
            self.logger.debug(f"Found {len(interactive_elements)} interactive elements")

            # Encode the activity (screen) to 1 hot
            activity_index = self.get_activity_index(current_activity)
            self.logger.debug(f"Activity index: {activity_index} (out of {self.max_activities})")
            activity_vector[activity_index] = 1
            
            semantic_vectors = []

            # Encode widgets
            self.logger.debug("Interactive elements details:")
            processed_widgets = []
            for i, element in enumerate(interactive_elements):
                widget_type = element.get('class', 'unknown') or element.get('className', 'unknown')
                widget_id = element.get('resource-id', 'unknown') or element.get('resourceId', 'unknown')
                widget_hash = self.hash_md5(widget_id + widget_type)

                widget_bounds = element.get('bounds', 'unknown') # [top,left] [botton,right]
                widget_coords = [0,0,0,0,0,0]
            
                widget_status = [
                    int(element.get('enabled', 'false').lower() == 'true'), 
                    int(element.get('checked', 'false').lower() == 'true'), 
                    int(element.get('password', 'false').lower() == 'true') 
                ]

                widget_events = [
                    int(element.get('clickable', 'false').lower() == 'true'),  
                    int(element.get('long-clickable', 'false').lower() == 'true' or element.get('longClickable', 'false').lower() == 'true'),  
                    int(element.get('scrollable', 'false').lower() == 'true'),  
                    int(element.get('focusable', 'false').lower() == 'true'),  
                    int(element.get('context-clickable', 'false').lower() == 'true')  
                ]
                widget_input_type = 'none' 

                # calculate center of bounds
                if widget_bounds != 'unknown':
                    bounds = widget_bounds.replace('][', ',').replace('[', '').replace(']', '').split(',')
                    if len(bounds) == 4:
                        top, left, bottom, right = map(int, bounds)
                        center_x = (left + right) / 2
                        center_y = (top + bottom) / 2
                        widget_coords = [top,left,bottom,right,center_x,center_y]
                
                # Map the widget to an index
                widget_index = self.get_widget_index(widget_hash)
                widget_vector[widget_index] = 1
                text_content = "TEXT:" + (element.get('text', '') + element.get('name',"")) + "/DESC:" + (element.get('content-desc', '')+ element.get('contentDescription', ''))
                if not text_content.strip():
                    text_content = "none"

                if 'EditText' in widget_type:
                    mapping_types = {
                        '1':'text',  # Default text input
                        '2':'number',
                        '3':'phone',
                        '4':'date-time',
                        '128':'password',
                        '32':'email',
                    }
                    if element.get('inputType', '1') in mapping_types:
                        widget_input_type = mapping_types[element.get('inputType', '1')]
                    
                    
                # Record widget details for logging
                widget_info = {
                    "id": widget_hash,
                    "type": widget_type,
                    "text": text_content, 
                    "input_type": widget_input_type,   
                    "status": widget_status, #array
                    "events":  widget_events,#array
                    "bounds": widget_coords, #array
                    "vector_index": widget_index
                }
                processed_widgets.append(widget_info)
                
                if text_content:
                    semantic_vectors.append(self.get_text_embedding(text_content))
            
            # Log processed widgets in a readable format
            if processed_widgets:
                self.logger.debug("Widget details :")
                for i, widget in enumerate(processed_widgets):
                    self.logger.debug(f"Widget {i}: {widget}")
                    
            orientation = self.get_device_orientation()
            network = self.get_network_status()
            focused_text = 1 if any(element.get('focused') == 'true' and 'EditText' in element.get('class', '') for element in root.findall(".//*")) else 0
            scrollable = 1 if any(element.get('scrollable') == 'true' for element in root.findall(".//*")) else 0
            additional_features = torch.tensor([orientation, network, focused_text, scrollable], dtype=torch.float32)
            self.logger.debug(f"Device features: orientation={orientation}, network={network}, focused_text={focused_text}, scrollable={scrollable}")

            # Combine into full state vector (add other features as needed)
            if semantic_vectors:
                semantic_vector = torch.mean(torch.stack(semantic_vectors), dim=0)
                self.logger.debug(f"Created semantic vector from {len(semantic_vectors)} text elements")
            else:
                semantic_vector = torch.zeros(self.semantic_dim)
                self.logger.debug("No text found, using zero semantic vector")
                
            state_vector = torch.cat((activity_vector, #500
                                      widget_vector,  #50
                                      additional_features,  #4 
                                      semantic_vector)) #384 
            
            self.logger.debug(f"Final state vector shape: {state_vector.shape}, non-zero elements: {torch.sum(state_vector != 0).item()}")
            return state_vector

        except Exception as e:
            self.logger.error(f"Error in GUI embedding: {str(e)}")
            return torch.zeros(self.state_dim)

    

    def visualize_state(self, state_vector: torch.Tensor, show_details: bool = False) -> str:
        """
        Creates a human-readable representation of the state vector.
        
        Args:
            state_vector: The state vector to visualize
            show_details: Whether to show the detailed breakdown of the vector
            
        Returns:
            A string representation of the state
        """
        # Extract the different components of the state vector
        activity_vector = state_vector[:self.max_activities]
        widget_vector = state_vector[self.max_activities:self.max_activities + self.max_widgets]
        additional_features = state_vector[self.max_activities + self.max_widgets:self.max_activities + self.max_widgets + 4]
        semantic_vector = state_vector[self.max_activities + self.max_widgets + 4:]
        
        # Find the active activity
        active_activity_idx = torch.argmax(activity_vector).item()
        active_activity = "unknown"
        for activity, idx in self.activity_dict.items():
            if idx == active_activity_idx:
                active_activity = activity
                break
        
        # Find active widgets
        active_widget_indices = torch.nonzero(widget_vector).flatten().tolist()
        active_widgets = []
        for widget_key, widget_idx in self.widget_dict.items():
            if widget_idx in active_widget_indices:
                widget_type, widget_text = widget_key
                active_widgets.append((widget_type, widget_text))
        
        # Get device state information
        orientation = "Portrait" if additional_features[0].item() == 0 else "Landscape"
        network = "On" if additional_features[1].item() == 1 else "Off"
        focused_text = "Yes" if additional_features[2].item() == 1 else "No"
        scrollable = "Yes" if additional_features[3].item() == 1 else "No"
        
        # Create a summary string
        summary = [
            f"Current Activity: {active_activity}",
            f"Interactive Elements: {len(active_widgets)}",
            f"Device State: Orientation={orientation}, Network={network}, Text Input Focus={focused_text}, Scrollable Content={scrollable}",
            f"Semantic Content: {'Present' if torch.sum(semantic_vector).item() > 0 else 'None'}"
        ]
        
        if show_details and active_widgets:
            summary.append("\nInteractive Elements:")
            for i, (widget_type, widget_text) in enumerate(active_widgets):
                type_name = widget_type.split('.')[-1] if '.' in widget_type else widget_type
                text_preview = widget_text[:30] + '...' if len(widget_text) > 30 else widget_text
                summary.append(f"  {i+1}. {type_name}: '{text_preview}'")
        
        return "\n".join(summary)