from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import re

bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)


class InputSuggestionLLM:
    def __init__(
        self,
        model_name="microsoft/Phi-4-mini-instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=bnb_config
        )
        self.model.eval()
        self.device = device

    def build_prompt(self, context: dict):
        return (
            "You are an expert in generating attack payload for android app input fields.\n"
            f"Generate 6 payloads total: 1 valid payload and 5 diverse, realistic and may cause error, edge case, invalid, vulnerability, injection payload.\n\n"
            "If no context is provided, generate text payloads for injections, errors, or edge cases.\n"
            f"Field metadata:\n"
            f"- Resource ID: {context.get('resource_id', '')}\n"
            f"- Content Description: {context.get('content_desc', '')}\n"
            f"- Payload Type: {context.get('input_type', '')}\n\n"
            f"Return ONLY RAW payload values as a list with tag <li>payload</li>. NO EXPLANATION\n"
            "OUTPUT FORMAT EXAMPLE: <li>payload</li>"
        )

    def suggest_inputs(self, context: dict, max_tokens=200):
        prompt = self.build_prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        suggestions = self.extract_suggestions(decoded[len(prompt) :])
        return suggestions

    @staticmethod
    def extract_suggestions(text):
        # Extract content between <li> and </li> tags using regex
        pattern = r"<li>(.*?)</li>"
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

        # Clean up the extracted content
        clean = []
        for match in matches:
            content = match.strip()
            if content:  # Only add non-empty content
                clean.append(content)

        # Fallback: if no <li> tags found, try the old method for backward compatibility
        if not clean:
            lines = text.strip().split("\n")
            for line in lines:
                if line.strip().startswith("-"):
                    line = line.strip("- ").strip()
                    if line:
                        clean.append(line)

        return clean


# context = {
#     "resource_id": "number_field",
#     "content_desc": "input username",
#     "input_type": "input_number",
# }

# generator = InputSuggestionLLM(model_name="microsoft/Phi-4-mini-instruct")
# suggestions = generator.suggest_inputs(context)

# print("LLM-generated input suggestions:")
# print(" ".join(suggestions))

# suggestions = generator.suggest_inputs(context)

# print("LLM-generated input suggestions:")
# print(" ".join(suggestions))

# suggestions = generator.suggest_inputs(context)

# print("LLM-generated input suggestions:")
# print(" ".join(suggestions))

# suggestions = generator.suggest_inputs(context)

# print("LLM-generated input suggestions:")
# print(" ".join(suggestions))
