# DeepInfra Python Client


## 🚨 Important Disclaimer

**EDUCATIONAL PURPOSE ONLY**

This code is provided **strictly for educational purposes only**. It is designed to demonstrate Python programming concepts including:
- Object-Oriented Programming
- Type Hints
- Error Handling
- API Integration
- Clean Code Practices

**Important Notes:**
- This code should not be used to harm, overwhelm, or abuse DeepInfra's services
- Always respect DeepInfra's terms of service and API usage guidelines
- For production use, please use DeepInfra's official API and documentation
- The authors take no responsibility for misuse of this code

## Overview

This Python client provides a clean, object-oriented interface for interacting with DeepInfra's AI models. It supports both text generation and vision-language tasks with various model options.

## Features

- Text generation with multiple model options
- Vision-language tasks support
- Streaming responses
- Configurable model parameters
- Type-safe interfaces
- Comprehensive error handling
- Clean, maintainable code structure

## Installation

### Prerequisites

```bash
pip install cloudscraper
```

### Required Python Version
- Python 3.7 or higher

## Usage

### Basic Text Generation

```python
from deepinfra import DeepInfra, ModelConfig

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
```

## Available Models

### Text Models
- Llama 3.1 405b
- Llama 3.3 70b
- Llama 3.1 8b
- Llama 3.3 70b Turbo
- Hermes Llama 3.1 405b
- DeepSeek v2.5
- QwQ 32B
- Mixtral 8x7b
- WizardLM2 8x22b
- WizardLM2 7b
- Qwen2.5 72b
- Gemma 2 27b
- Mistral 7b v0.3
- openchat 3.5 7b
- MythoMax 13b
- Qwen2.5 Coder 32b
- Nemotron3.1 70B

### Vision Models
- Llama-3.2-11B-Vision-Instruct
- Llama-3.2-90B-Vision-Instruct

## Configuration

The `ModelConfig` class allows customization of model behavior:

```python
config = ModelConfig(
    max_tokens=2048,    # Maximum tokens to generate
    temperature=0.7,    # Controls randomness (0-1)
    top_p=0.7,         # Nucleus sampling parameter (0-1)
    system_prompt="Be a helpful assistant"  # System behavior prompt
)
```

## Error Handling

The client includes custom exceptions for better error handling:

```python
try:
    response = client.text.generate("Hello", model="Invalid Model")
except ModelNotFoundError as e:
    print(f"Model error: {e}")
except APIError as e:
    print(f"API error: {e}")
except DeepInfraError as e:
    print(f"General error: {e}")
```

## Contributing

While this is an educational project, suggestions for improving the code structure, documentation, or educational value are welcome. Please ensure any contributions:

1. Follow the existing code style
2. Include appropriate type hints
3. Add proper documentation
4. Respect the educational purpose of the project

## Acknowledgments

- This project is created for educational purposes to demonstrate Python programming concepts
- Thanks to DeepInfra for providing amazing AI models
- This is not an official DeepInfra project

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For questions or support, please open an issue or reach out to the maintainer.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the project repository.
