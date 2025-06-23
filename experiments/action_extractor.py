from typing import List, Dict, Any
from lxml import etree as ET
from appium.webdriver.webdriver import WebDriver
from experiments.logger import setup_logger
import datetime
import torch

log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class ActionExtractor:
    def __init__(self, driver: WebDriver, log_dir: str = "app/logs/", emulator_name: str = "Unknown", app_name: str = "Unknown"):
        self.driver = driver
        self.logger = setup_logger(f"{log_dir}/action_extractor.log", emulator_name=emulator_name, app_name=app_name)
        self.app_name = app_name
        self.text_dictionary = ["test", "123", "hello", "android", "input", "-54.67", "20.00", "00.1","-2", "232141234124","-232141234124","a@a.a"]  # Example texts
        self.system_actions = [
            {'type': 'system', 'action': 'toggle_network'},
            {'type': 'system', 'action': 'rotate_screen'}
        ]

    def extract_actions(self, gui_hierarchy: str, widget_dict: Dict) -> tuple[List[Dict[str, Any]], List[torch.Tensor]]:
        if widget_dict is None:
            self.logger.error("widget_dict is None in extract_actions")
            return [], []
        
        # Check if widget_dict is a dictionary
        if not isinstance(widget_dict, dict):
            self.logger.error(f"widget_dict is not a dictionary, but {type(widget_dict)}")
            return [], []
        
        # Log widget_dict details for debugging
        #self.logger.debug(f"Type of widget_dict: {type(widget_dict)}, length: {len(widget_dict)}")
        
        actions = []
        action_vectors = []
        
        root = ET.fromstring(gui_hierarchy.encode('utf-8'))

        clickable_elements = root.findall(".//*[@clickable='true']")
        for element in clickable_elements:
            widget_key = (element.get('class'), element.get('text', ''))
            if widget_key in widget_dict:  
                widget_index = widget_dict[widget_key]
                action = {'type': 'touch', 'widget_index': widget_index, 'parameters': None}
                actions.append(action)
                action_vectors.append(self.action_to_vector(action))

        # Extract scrollable actions (gestures)
        scrollable_elements = root.findall(".//*[@scrollable='true']")
        for element in scrollable_elements:
            widget_key = (element.get('class'), element.get('text', ''))
            if widget_key in widget_dict:
                widget_index = widget_dict[widget_key]
                for direction in ['up', 'down', 'left', 'right']:
                    action = {'type': 'gesture', 'widget_index': widget_index, 'parameters': {'gesture_type': 'swipe', 'direction': direction}}
                    actions.append(action)
                    action_vectors.append(self.action_to_vector(action))

        # Extract editable actions (text inputs)
        # editable_elements = root.findall(".//*[@class='android.widget.EditText']")
        # for element in editable_elements:
        #     widget_key = (element.get('class'), element.get('text', ''))
        #     if widget_key in widget_dict:
        #         widget_index = widget_dict[widget_key]  # Fixed: Corrected widget_dict indexing
        #         for text_idx, text in enumerate(self.text_dictionary):
        #             action = {'type': 'text_input', 'widget_index': widget_index, 'parameters': {'text': text, 'text_idx': text_idx}}
        #             actions.append(action)
        #             action_vectors.append(self.action_to_vector(action))
        editable_elements = root.findall(".//*[@class='android.widget.EditText']")
        for element in editable_elements:
            widget_key = (element.get('class'), element.get('text', ''))
            if widget_key in widget_dict:
                widget_index = widget_dict[widget_key]

                # Add clear_text before any new input
                clear_action = {'type': 'clear_text', 'widget_index': widget_index}
                actions.append(clear_action)
                action_vectors.append(self.action_to_vector(clear_action))

                # Add text input actions
                for text_idx, text in enumerate(self.text_dictionary):
                    action = {
                        'type': 'text_input',
                        'widget_index': widget_index,
                        'parameters': {'text': text, 'text_idx': text_idx}
                    }
                    actions.append(action)
                    action_vectors.append(self.action_to_vector(action))

        # Add confirm_input buttons (rule-based for now)
        possible_confirm_buttons = root.findall(".//*[@clickable='true']")
        confirm_keywords = {"ok", "submit", "confirm", "done", "send"}
        for element in possible_confirm_buttons:
            button_text = element.get('text', '').strip().lower()
            if button_text in confirm_keywords:
                widget_key = (element.get('class'), element.get('text', ''))
                if widget_key in widget_dict:
                    widget_index = widget_dict[widget_key]
                    confirm_action = {'type': 'confirm_input', 'widget_index': widget_index}
                    actions.append(confirm_action)
                    action_vectors.append(self.action_to_vector(confirm_action))

        # Add system actions
        # for system_action in self.system_actions:
        #     actions.append(system_action)
        #     action_vectors.append(self.action_to_vector(system_action))
        # Context-aware system actions
        gui_text = ET.tostring(root, encoding='unicode').lower()

        # Include network toggle only if spinner or offline-indicator exists
        if any(keyword in gui_text for keyword in ["loading", "offline", "try again", "no internet", "progressbar"]):
            net_action = {'type': 'system', 'action': 'toggle_network'}
            actions.append(net_action)
            action_vectors.append(self.action_to_vector(net_action))

        # Include rotate only if text fields are present
        edit_fields = root.findall(".//*[@class='android.widget.EditText']")
        if len(edit_fields) > 0:
            rot_action = {'type': 'system', 'action': 'rotate_screen'}
            actions.append(rot_action)
            action_vectors.append(self.action_to_vector(rot_action))


        # === Detect Pop-up Dialog and Add `touch_outside` Action ===
        modal_root = root.find(".//*[@class='android.app.Dialog']")
        has_clickables = root.findall(".//*[@clickable='true']")

        if modal_root is not None and not has_clickables:
            bounds_str = modal_root.get("bounds", "[0,0][100,100]")  # fallback bounds
            try:
                x1, y1, x2, y2 = map(int, bounds_str.replace('[', '').replace(']', ',').split(',')[:4])
                # Tap slightly outside bottom-right of the dialog
                outside_x = x2 + 20
                outside_y = y2 + 20
                touch_outside_action = {
                    'type': 'touch_outside',
                    'parameters': {'x': outside_x, 'y': outside_y}
                }
                actions.append(touch_outside_action)
                action_vectors.append(self.action_to_vector(touch_outside_action))
            except Exception as e:
                self.logger.warning(f"Failed to parse dialog bounds: {e}")



        #self.logger.info(f"Extracted {len(actions)} actions with corresponding action vectors")
        return actions, action_vectors
        # except Exception as e:
        #     self.logger.error(f"Error extracting actions: {str(e)}")
        #     return actions, action_vectors
        
    def _parse_bounds(self, bounds: str) -> tuple[int, int]:
        """Parse bounds string to get center coordinates."""
        try:
            x1, y1, x2, y2 = map(int, bounds.replace('[', '').replace(']', ',').split(',')[:4])
            return (x1 + x2) // 2, (y1 + y2) // 2
        except:  # noqa: E722
            return 0, 0
    
    def _generate_xpath(self, element: ET.Element) -> str:
        """
        Generate an XPath for an element if not provided.
        Uses tag, index, and resource-id (if available) to create a unique path.
        """
        tag = element.tag
        resource_id = element.get('resource-id', '')
        index = element.get('index', '0')

        if resource_id:
            return f"//*[@resource-id='{resource_id}']"
        else:
            # Fallback to tag and index
            parent = element.getparent()
            if parent is None:
                return f"//{tag}[{index}]"
            siblings = parent.findall(f".//{tag}")
            idx = siblings.index(element) + 1
            return f"//{tag}[{idx}]"

    def _is_element_interactable(self, xpath: str) -> bool:
        """
        Verify if an element is interactable (visible and enabled).
        Returns False if the element cannot be found or is not interactable.
        """
        try:
            element = self.driver.find_element(by='xpath', value=xpath)
            return element.is_displayed() and element.is_enabled()
        except Exception as e:
            self.logger.debug(f"Element at {xpath} is not interactable: {str(e)}")
            return False
    
    # def action_to_vector(self, action: Dict[str, Any], max_widgets: int = 500, text_dict_size: int = 10, system_action_size: int = 2) -> torch.Tensor:
    #     action_type_onehot = torch.zeros(4)  # touch, gesture, text_input, system
    #     target_index = -1
    #     gesture_type_onehot = torch.zeros(4)  # e.g., swipe, pinch, zoom_in, zoom_out
    #     direction_onehot = torch.zeros(4)  # up, down, left, right
    #     text_index = -1
    #     system_id = -1

    #     if action['type'] == 'touch':
    #         action_type_onehot[0] = 1
    #         target_index = action['widget_index'] / max_widgets
    #     elif action['type'] == 'gesture':
    #         action_type_onehot[1] = 1
    #         target_index = action['widget_index'] / max_widgets
    #         gesture_type = action['parameters']['gesture_type']
    #         if gesture_type == 'swipe':
    #             gesture_type_onehot[0] = 1
    #         # Add other gesture types as needed
    #         direction = action['parameters']['direction']
    #         if direction == 'up':
    #             direction_onehot[0] = 1
    #         elif direction == 'down':
    #             direction_onehot[1] = 1
    #         elif direction == 'left':
    #             direction_onehot[2] = 1
    #         elif direction == 'right':
    #             direction_onehot[3] = 1
    #     elif action['type'] == 'text_input':
    #         action_type_onehot[2] = 1
    #         target_index = action['widget_index'] / max_widgets
    #         text_index = action['parameters']['text_idx'] / text_dict_size
    #     elif action['type'] == 'system':
    #         action_type_onehot[3] = 1
    #         system_action = action['action']
    #         if system_action == 'toggle_network':
    #             system_id = 0 / system_action_size
    #         elif system_action == 'rotate_screen':
    #             system_id = 1 / system_action_size

    #     action_vector = torch.cat((
    #         action_type_onehot,
    #         torch.tensor([target_index]),
    #         gesture_type_onehot,
    #         direction_onehot,
    #         torch.tensor([text_index]),
    #         torch.tensor([system_id])
    #     ))
    #     return action_vector
    def action_to_vector(self, action: dict, max_widgets: int = 500, text_dict_size: int = 10, system_action_size: int = 2) -> torch.Tensor:
        # One-hot encode action type (touch, gesture, text_input, system)
        action_type_onehot = torch.zeros(4)
        widget_index = action.get('widget_index', -1) / max_widgets if action.get('widget_index') is not None else 0
        gesture_type_onehot = torch.zeros(4)  # swipe, pinch, zoom_in, zoom_out
        direction_onehot = torch.zeros(4)  # up, down, left, right
        text_index = 0
        system_id = 0

        # Get parameters, default to empty dict if None or missing
        parameters = action.get('parameters', {}) or {}

        # Fill in the vector based on action type
        if action['type'] == 'touch':
            action_type_onehot[0] = 1
        elif action['type'] == 'gesture':
            action_type_onehot[1] = 1
            gesture_type = parameters.get('gesture_type', '')
            if gesture_type == 'swipe':
                gesture_type_onehot[0] = 1
            elif gesture_type == 'pinch':
                gesture_type_onehot[1] = 1
            direction = parameters.get('direction', '')
            if direction == 'up':
                direction_onehot[0] = 1
            elif direction == 'down':
                direction_onehot[1] = 1
            elif direction == 'left':
                direction_onehot[2] = 1
            elif direction == 'right':
                direction_onehot[3] = 1
        elif action['type'] == 'text_input':
            action_type_onehot[2] = 1
            text_index = parameters.get('text_idx', -1) / text_dict_size
        elif action['type'] == 'system':
            action_type_onehot[3] = 1
            system_action = action.get('action', '')
            if system_action == 'toggle_network':
                system_id = 0 / system_action_size
            elif system_action == 'rotate_screen':
                system_id = 1 / system_action_size
        elif action['type'] == 'touch_outside':
            action_type_onehot[0] = 1  # reuse 'touch'
            # normalize coordinates roughly to 0-1 scale if needed
            widget_index = 0.99  # or use x/y if you later add spatial vector


        # Combine into a fixed-size vector
        action_vector = torch.cat((
            action_type_onehot,           # 4 elements
            torch.tensor([widget_index]), # 1 element
            gesture_type_onehot,          # 4 elements
            direction_onehot,             # 4 elements
            torch.tensor([text_index]),   # 1 element
            torch.tensor([system_id]),     # 1 element
            #torch.zeros(49)                # Placeholder for future features (e.g., bounds, xpath)
        ))  # Total: 15 elements 

        return action_vector