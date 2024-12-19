from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import base64
import cloudscraper

@dataclass
class ModelConfig:
    """Configuration settings for model inference.

    Attributes:
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature between 0 and 1
        top_p (float): Nucleus sampling parameter between 0 and 1
        system_prompt (str): System prompt to guide model behavior
    """
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.7
    system_prompt: str = 'Be a helpful assistant'

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be greater than 0")

class DeepInfraError(Exception):
    """Base exception class for DeepInfra-related errors."""
    pass

class ModelNotFoundError(DeepInfraError):
    """Exception raised when specified model is not available."""
    pass

class APIError(DeepInfraError):
    """Exception raised for API-related errors."""
    pass

class Message:
    """Represents a chat message with role and content."""
    
    def __init__(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Initialize a chat message.

        Args:
            role (str): Role of the message sender (system/user/assistant)
            content (Union[str, List[Dict[str, Any]]]): Message content
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API."""
        return {
            'role': self.role,
            'content': self.content
        }

class ModelRegistry:
    """Registry maintaining available models and their endpoints."""

    TEXT_MODELS = {
        "Llama 3.1 405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "Llama 3.3 70b": "meta-llama/Llama-3.3-70B-Instruct",
        "Llama 3.1 8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Llama 3.3 70b Turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Hermes Llama 3.1 405b": "NousResearch/Hermes-3-Llama-3.1-405B",
        "DeepSeek v2.5": "deepseek-ai/DeepSeek-V2.5",
        "QwQ 32B": "Qwen/QwQ-32B-Preview",
        "Mixtral 8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "WizardLM2 8x22b": "microsoft/WizardLM-2-8x22B",
        "WizardLM2 7b": "microsoft/WizardLM-2-7B",
        "Qwen2.5 72b": "Qwen/Qwen2.5-72B-Instruct",
        "Gemma 2 27b": "google/gemma-2-27b-it",
        "Mistral 7b v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "openchat 3.5 7b": "openchat/openchat_3.5",
        "MythoMax 13b": "Gryphe/MythoMax-L2-13b",
        "Qwen2.5 Coder 32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Nemotron3.1 70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct"
    }

    VISION_MODELS = {
        'Llama-3.2-11B-Vision-Instruct': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'Llama-3.2-90B-Vision-Instruct': 'meta-llama/Llama-3.2-90B-Vision-Instruct'
    }

    @classmethod
    def get_model_endpoint(cls, model_name: str, vision: bool = False) -> str:
        """
        Get the API endpoint for a given model name.

        Args:
            model_name (str): Name of the model
            vision (bool): Whether to check vision models

        Returns:
            str: Model endpoint

        Raises:
            ModelNotFoundError: If model is not found in registry
        """
        models = cls.VISION_MODELS if vision else cls.TEXT_MODELS
        if model_name not in models:
            available = list(models.keys())
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. Available models: {', '.join(available)}"
            )
        return models[model_name]

class BaseInference(ABC):
    """Abstract base class for model inference."""

    def __init__(self, timeout: int = 30, prints: bool = True) -> None:
        """
        Initialize the inference client.

        Args:
            timeout (int): Request timeout in seconds
            prints (bool): Whether to print streaming responses
        """
        self.timeout = timeout
        self.prints = prints
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'Sec-GPC': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Brave";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }
        self.scraper = cloudscraper.create_scraper()

    @abstractmethod
    def prepare_messages(self, *args, **kwargs) -> List[Message]:
        """Prepare messages for API request."""
        pass

    def _make_request(self, messages: List[Message], model: str, config: ModelConfig) -> str:
        """
        Make API request and handle streaming response.

        Args:
            messages (List[Message]): List of prepared messages
            model (str): Model endpoint
            config (ModelConfig): Model configuration

        Returns:
            str: Generated response

        Raises:
            APIError: If API request fails
        """
        json_data = {
            'model': model,
            'messages': [msg.to_dict() for msg in messages],
            'stream': True,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
        }

        try:
            response = self.scraper.post(
                'https://api.deepinfra.com/v1/openai/chat/completions',
                headers=self.headers,
                json=json_data,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            if response.status_code == 200:
                return self._handle_streaming_response(response)
            else:
                raise APIError(f"API request failed: {response.status_code} - {response.reason}")

        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")

    def _handle_streaming_response(self, response) -> str:
        """
        Handle streaming response from API.

        Args:
            response: Response object from API

        Returns:
            str: Concatenated response content
        """
        streaming_response = ''
        for value in response.iter_lines(decode_unicode=True, chunk_size=1000):
            if value and value.startswith("data:"):
                try:
                    json_str = value[5:]
                    parsed_json = json.loads(json_str)
                    content = parsed_json["choices"][0]["delta"]["content"]
                    streaming_response += content
                    if self.prints:
                        print(content, end="", flush=True)
                except:
                    continue
        return streaming_response

class TextGeneration(BaseInference):
    """Handle text generation inference."""

    def prepare_messages(self, query: str, config: ModelConfig) -> List[Message]:
        """
        Prepare messages for text generation.

        Args:
            query (str): User query
            config (ModelConfig): Model configuration

        Returns:
            List[Message]: Prepared messages
        """
        return [
            Message('system', config.system_prompt),
            Message('user', query)
        ]

    def generate(self, query: str, model: str = "Llama 3.3 70b Turbo", config: Optional[ModelConfig] = None) -> str:
        """
        Generate text response for given query.

        Args:
            query (str): User query
            model (str): Model name
            config (Optional[ModelConfig]): Model configuration

        Returns:
            str: Generated response

        Raises:
            ModelNotFoundError: If model is not found
            APIError: If API request fails
        """
        config = config or ModelConfig()
        model_endpoint = ModelRegistry.get_model_endpoint(model)
        messages = self.prepare_messages(query, config)
        return self._make_request(messages, model_endpoint, config)

class VisionGeneration(BaseInference):
    """Handle vision-language inference."""

    def prepare_messages(self, query: str, image_path: str, config: ModelConfig) -> List[Message]:
        """
        Prepare messages for vision-language tasks.

        Args:
            query (str): User query
            image_path (str): Path to image file
            config (ModelConfig): Model configuration

        Returns:
            List[Message]: Prepared messages
        """
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()

        content = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{image_base64}'
                }
            },
            {
                'type': 'text',
                'text': query
            }
        ]

        return [
            Message('system', config.system_prompt),
            Message('user', content)
        ]

    def generate(self, query: str, image_path: str, 
                model: str = "Llama-3.2-11B-Vision-Instruct",
                config: Optional[ModelConfig] = None) -> str:
        """
        Generate response for image-based query.

        Args:
            query (str): User query about the image
            image_path (str): Path to image file
            model (str): Model name
            config (Optional[ModelConfig]): Model configuration

        Returns:
            str: Generated response

        Raises:
            ModelNotFoundError: If model is not found
            APIError: If API request fails
        """
        config = config or ModelConfig()
        model_endpoint = ModelRegistry.get_model_endpoint(model, vision=True)
        messages = self.prepare_messages(query, image_path, config)
        return self._make_request(messages, model_endpoint, config)

class DeepInfra:
    """Main client class for DeepInfra API."""

    def __init__(self, timeout: int = 30, prints: bool = True) -> None:
        """
        Initialize DeepInfra client.

        Args:
            timeout (int): Request timeout in seconds
            prints (bool): Whether to print streaming responses
        """
        self.text = TextGeneration(timeout, prints)
        self.vision = VisionGeneration(timeout, prints)

if __name__ == "__main__":
    try:
        # Example usage
        client = DeepInfra()
        
        # Text generation example
        config = ModelConfig(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.7,
            system_prompt="You are a helpful assistant."
        )
        
        text_response = client.text.generate(
            "Hi, how are you?",
            model="Llama 3.3 70b Turbo",
            config=config
        )
        print(f'\n\nText Generation Response: {text_response}')
        
        # Vision generation example
        vision_response = client.vision.generate(
            query="Describe this image",
            image_path="image.jpg",
            model="Llama-3.2-11B-Vision-Instruct",
            config=config
        )
        print(f'\n\nImage To Text Response: {vision_response}')
        
    except DeepInfraError as e:
        print(f"DeepInfra Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")