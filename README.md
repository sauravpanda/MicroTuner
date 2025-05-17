# MicroTuner 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/Modal-Deploy-black)](https://modal.com/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-orange)](https://github.com/unslothai/unsloth)

Fine-tune lightweight LLMs in minutes, not hours. MicroTuner leverages [Modal](https://modal.com/) for serverless deployment and [Unsloth](https://github.com/unslothai/unsloth) for optimized training, making it easy to create your own specialized models without massive computational resources.

## 🌟 Features

- **Cloud-Native**: Built on Modal for seamless cloud deployment and GPU access
- **Ultra-Fast**: Uses Unsloth's optimized training for 2-5x speedup over traditional methods
- **Flexible**: Fine-tune any Llama or Mistral-based model for your specific task
- **Production-Ready**: Automatically publishes your model to Hugging Face Hub
- **Resource-Efficient**: 4-bit quantization allows training on consumer GPUs
- **Minimal Setup**: One command to deploy and train

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/microtuner.git
cd microtuner
```

2. Install Poetry if you don't have it:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Set up Modal:
```bash
poetry run modal token new
```

## 💻 Usage

### Setup your HuggingFace token

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Run training

```bash
poetry run modal run train.py \
    --base-model="unsloth/Llama-3.2-1B" \
    --dataset-name="huggingface-your-username/your-dataset" \
    --output-repo="huggingface-your-username/your-model-name" \
    --num-train-samples=300000
```

### Sample Command

Here's a complete example of fine-tuning a small SQL generation model:

```bash
poetry run modal run train.py \
    --base-model="unsloth/Llama-3.2-1B" \
    --dataset-name="cloudcodeai/sqlqa-test" \
    --output-repo="cloudcodeai/sqlqa-test-llama-3-1b" \
    --num-train-samples=3000
```

## 📋 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base-model` | The base model to fine-tune | `unsloth/Llama-3.2-1B` |
| `--dataset-name` | HuggingFace dataset name | `PowerInfer/QWQ-LONGCOT-500K` |
| `--output-repo` | Where to push the fine-tuned model | Required |
| `--num-train-samples` | Number of training samples to use | 300000 |
| `--evaluation-samples` | Number of samples for evaluation | 50 |
| `--num-train-epochs` | Number of training epochs | 2 |
| `--learning-rate` | Learning rate for training | 2e-4 |
| `--task-type` | Type of task (chat, completion, classification) | chat |
| `--input-field` | Field in dataset to use as input | auto-detect |
| `--output-field` | Field in dataset to use as output | auto-detect |

## 📊 Dataset Format

The code automatically adapts to various dataset formats. However, the most straightforward format is:

```json
{
  "question": "Your input text here",
  "answer": "Your expected output here"
}
```

You can specify custom field names using the `--input-field` and `--output-field` parameters.

## ⚡ What Happens Behind the Scenes

When you run the command:

1. Modal provisions a cloud environment with the necessary GPU
2. The script downloads the base model and loads it with Unsloth's optimizations
3. Your dataset is loaded from Hugging Face and pre-processed
4. The model is fine-tuned using QLoRA for efficient training
5. If specified, the fine-tuned model is pushed to your Hugging Face repository

All computation happens in the cloud, so you don't need a powerful local machine!

## 🔄 Common Use Cases

MicroTuner excels at creating specialized models for:

- **Text classification**: Train models to categorize text into predefined classes
- **Content generation**: Create models that generate specific types of content
- **Instruction following**: Fine-tune models to follow domain-specific instructions
- **Question answering**: Build models that excel at answering questions in your domain
- **SQL generation**: Create models that convert natural language to SQL queries
- **Conversational agents**: Create chatbots tailored to specific industries

## 🧪 Example Usage After Fine-Tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("yourusername/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("yourusername/your-model-name")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1
)

prompt = "Your prompt here..."
result = generator(prompt)
print(result[0]['generated_text'])
```

## 🛠️ Advanced Configuration

For advanced users, the script offers many configuration options:

```bash
poetry run modal run train.py --help
```

You can customize:

- LoRA parameters (rank, alpha, target modules)
- Training hyperparameters (batch size, learning rate, weight decay)
- Model configuration (sequence length, quantization type)
- Chat template formatting

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{microtuner,
  author = {Saurav Panda},
  title = {MicroTuner: Fast LLM Fine-Tuning with Modal and Unsloth},
  year = {2025},
  url = {https://github.com/sauravpanda/microtuner}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- [Modal](https://modal.com/) for their excellent serverless compute platform
- [Unsloth](https://github.com/unslothai/unsloth) for their optimized fine-tuning library
- [Hugging Face](https://huggingface.co/) for hosting models and datasets
