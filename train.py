from modal import Image, App, method
from dataclasses import dataclass
import os
from typing import Optional, Dict, Any, List

# Constants
MAX_NEW_TOKENS = 512
GENERATION_TEMPERATURE = 0.1
DEFAULT_FIELDS = {
    "input": ["input", "question", "prompt"],
    "output": ["output", "answer", "response"],
    "label": ["label", "class", "category"]
}
GPU = "L4" # Change this based on modal GPU types

def download_deps():
    import subprocess
    # subprocess.run(["pip", "uninstall", "unsloth", "-y"])  # First remove existing unsloth
    subprocess.run(["pip", "install", "unsloth"])

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "accelerate>=0.25.0", 
        "bitsandbytes>=0.41.0",
        "transformers>=4.37.2",
        "datasets",
        "peft>=0.7.0",
        "huggingface_hub",
        "scipy",
        "ninja",
    )
    .run_function(download_deps)
)

app = App("microtuner", image=image)

@dataclass
class TrainingConfig:
    base_model: str
    dataset_name: str
    output_repo: str
    hf_token: str
    # Dataset config
    num_train_samples: int = 300000
    input_field: Optional[str] = None  # Field to use as input
    output_field: Optional[str] = None  # Field to use as output
    # Model config
    max_seq_length: int = 32000
    load_in_4bit: bool = True
    chat_template: str = "llama-3.1"  # Template to use
    # Training config
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # Evaluation
    evaluation_samples: int = 50  # Number of samples to use for validation
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0
    # Task type
    task_type: str = "chat"  # Options: "chat", "completion", "classification"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_train_samples <= 0:
            raise ValueError("num_train_samples must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.task_type not in ["chat", "completion", "classification"]:
            raise ValueError("Invalid task_type")

@app.cls(gpu=GPU, timeout=24 * 60 * 60)
class ModelTrainer:
    # Initialize dtype at class level
    dtype = None  # Auto detection

    def create_or_get_repo(self, repo_name: str, token: str) -> None:
        """Create a new model repository on HuggingFace if it doesn't exist.
        
        Args:
            repo_name: Name of the repository to create/check
            token: HuggingFace API token
            
        Raises:
            ValueError: If the token is invalid
            HTTPError: If there's an issue with the HuggingFace API
        """
        from huggingface_hub import create_repo, HfApi
        api = HfApi()
        
        try:
            api.repo_info(repo_id=repo_name, token=token)
            print(f"Repository {repo_name} already exists.")
        except Exception as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid HuggingFace token") from e
            elif e.response.status_code == 404:
                create_repo(
                    repo_id=repo_name,
                    token=token,
                    private=True,
                    exist_ok=True
                )
                print(f"Created new repository: {repo_name}")
            else:
                raise

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: List[Dict[str, Any]],
        config: TrainingConfig
    ) -> List[Dict[str, str]]:
        """Evaluate the model's generation capability.
        
        Args:
            model: The trained model
            tokenizer: The model's tokenizer
            eval_dataset: Dataset to evaluate on
            config: Training configuration
            
        Returns:
            List of dictionaries containing prompt, reference, and generated text
        """
        import torch
        from tqdm import tqdm
        
        results = []
        
        for example in tqdm(eval_dataset, desc="Evaluating model"):
            if config.task_type == "chat":
                prompt = example["conversations"][0]["content"]
                reference = example["conversations"][1]["content"] if len(example["conversations"]) > 1 else ""
            else:
                prompt = example.get(config.input_field, "")
                reference = example.get(config.output_field, "")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if config.task_type == "chat" and config.chat_template == "llama-3.1":
                assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                if assistant_prefix in generated_text:
                    generated_text = generated_text.split(assistant_prefix)[1]
            
            results.append({
                "prompt": prompt,
                "reference": reference,
                "generated": generated_text
            })
        
        return results

    def get_preprocessing_function(self, config):
        """Returns the appropriate preprocessing function based on the task type."""
        if config.task_type == "chat":
            # For chat/conversation tasks
            def preprocess_conversation(example):
                # Handle various dataset formats
                if config.input_field and config.output_field:
                    # Use specified fields
                    return {
                        "conversations": [
                            {"role": "human", "content": example[config.input_field]},
                            {"role": "gpt", "content": example[config.output_field]}
                        ]
                    }
                elif "conversations" in example:
                    # Dataset already has conversations
                    return example
                elif "messages" in example:
                    # Convert messages format to conversations
                    return {
                        "conversations": [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in example["messages"]
                        ]
                    }
                else:
                    # Default to best guess of fields
                    input_field = config.input_field or next((f for f in example.keys() if f in DEFAULT_FIELDS["input"]), list(example.keys())[0])
                    output_field = config.output_field or next((f for f in example.keys() if f in DEFAULT_FIELDS["output"]), list(example.keys())[1] if len(example.keys()) > 1 else None)
                    
                    return {
                        "conversations": [
                            {"role": "human", "content": example[input_field]},
                            {"role": "gpt", "content": example[output_field] if output_field else ""}
                        ]
                    }
            return preprocess_conversation
        
        elif config.task_type == "completion":
            # For text completion tasks
            def preprocess_completion(example):
                # Just pass the raw text
                if config.input_field:
                    return {"text": example[config.input_field]}
                elif "text" in example:
                    return example
                else:
                    # Use the first field we find
                    return {"text": example[list(example.keys())[0]]}
            return preprocess_completion
        
        elif config.task_type == "classification":
            # For classification tasks
            def preprocess_classification(example):
                input_field = config.input_field or next((f for f in example.keys() if f in DEFAULT_FIELDS["input"]), list(example.keys())[0])
                label_field = config.output_field or next((f for f in example.keys() if f in DEFAULT_FIELDS["label"]), list(example.keys())[1] if len(example.keys()) > 1 else None)
                
                return {
                    "conversations": [
                        {"role": "human", "content": example[input_field]},
                        {"role": "gpt", "content": str(example[label_field]) if label_field else ""}
                    ]
                }
            return preprocess_classification
        
        else:
            # Default fallback
            def preprocess_default(example):
                return example
            return preprocess_default

    @method()
    def train(self, config: TrainingConfig):
        """Train an LLM model for using the Unsloth library."""
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments, DataCollatorForSeq2Seq
        from unsloth import is_bfloat16_supported
        from unsloth.chat_templates import train_on_responses_only, standardize_sharegpt
        import logging
        import time
        from tqdm import tqdm
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        logger.info(f"Starting training process with {config.base_model}")
        logger.info(f"Task type: {config.task_type}")
        
        # Create or get HuggingFace repository
        self.create_or_get_repo(config.output_repo, config.hf_token)
        
        # Load base model
        logger.info(f"Loading base model: {config.base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=config.load_in_4bit,
            token=config.hf_token
        )
        logger.info("Base model loaded successfully")

        # Add LoRA adapters
        logger.info("Adding LoRA adapters to the model")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Setup tokenizer with chat template
        from unsloth.chat_templates import get_chat_template
        logger.info(f"Configuring tokenizer with {config.chat_template} chat template")
        if config.task_type in ["chat", "classification"]:
            # Only apply chat template for chat or classification tasks
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=config.chat_template,
            )

        # Load dataset
        logger.info(f"Loading dataset: {config.dataset_name} with {config.num_train_samples} samples")
        try:
            dataset = load_dataset(
                config.dataset_name,
                split=f"train[:{config.num_train_samples}]",
                token=config.hf_token
            )
            logger.info(f"Dataset loaded with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Split dataset for training and evaluation
        if config.evaluation_samples > 0:
            eval_size = min(config.evaluation_samples, len(dataset) // 10)
            split_dataset = dataset.train_test_split(test_size=eval_size, seed=42)
            train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
            logger.info(f"Split dataset: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")
        else:
            train_dataset = dataset
            eval_dataset = None

        # Get preprocessing function based on task type
        preprocess_function = self.get_preprocessing_function(config)
        
        logger.info("Preprocessing training dataset")
        train_dataset = train_dataset.map(
            preprocess_function, 
            remove_columns=train_dataset.column_names
        )
        
        if config.task_type == "chat":
            train_dataset = standardize_sharegpt(train_dataset)
        
        if eval_dataset:
            logger.info("Preprocessing evaluation dataset")
            eval_dataset = eval_dataset.map(
                preprocess_function, 
                remove_columns=eval_dataset.column_names
            )
            if config.task_type == "chat":
                eval_dataset = standardize_sharegpt(eval_dataset)
        
        # Format prompts appropriately based on task type
        if config.task_type == "chat":
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                        for convo in convos]
                return {"text": texts}
            
            logger.info("Formatting chat prompts")
            processed_train_dataset = train_dataset.map(
                formatting_prompts_func, 
                batched=True,
                num_proc=2
            )
        elif config.task_type == "completion":
            # For completion, we already have text field
            processed_train_dataset = train_dataset
        else:
            # For classification, similar to chat
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                        for convo in convos]
                return {"text": texts}
            
            logger.info("Formatting classification prompts")
            processed_train_dataset = train_dataset.map(
                formatting_prompts_func, 
                batched=True,
                num_proc=2
            )

        # Initialize trainer
        logger.info("Initializing SFT trainer")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=processed_train_dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=config.num_train_epochs,
                max_steps=-1,
                learning_rate=config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )

        # Only use train_on_responses_only for chat/classification tasks
        if config.task_type in ["chat", "classification"] and config.chat_template == "llama-3.1":
            logger.info("Configuring training on responses only")
            trainer = train_on_responses_only(
                trainer,
                instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
            
            # Before training verification for chat/classification
            logger.info("Verifying mask before training...")
            if len(trainer.train_dataset) > 5:
                logger.info(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
                space = tokenizer(" ", add_special_tokens = False).input_ids[0]
                logger.info(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))
        
        # Train the model
        logger.info("Starting training")
        try:
            trainer_stats = trainer.train()
            logger.info(f"Training completed - Loss: {trainer_stats.training_loss}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        logger.info("Pushing model to Hugging Face Hub")
        model.push_to_hub_merged(
            config.output_repo,
            tokenizer,
            save_method = "merged_16bit",
            token = config.hf_token
        )

        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Training process completed in {elapsed_time:.2f} minutes")
        logger.info(f"Model successfully pushed to {config.output_repo}")
        
        return {
            "training_stats": trainer_stats.metrics,
            "elapsed_minutes": elapsed_time
        }

@app.local_entrypoint()
def main(
    base_model: str = "unsloth/Llama-3.2-1B",
    dataset_name: str = "PowerInfer/QWQ-LONGCOT-500K",
    output_repo: str = None,
    num_train_samples: int = 300000,
    task_type: str = "chat",
    input_field: str = None,
    output_field: str = None,
    evaluation_samples: int = 50,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    chat_template: str = "llama-3.1",
    lora_r: int = 32,
    hf_token: str = None,
):
    """
    Fine-tune a language model with Unsloth and Modal.
    
    Args:
        base_model: Base model to fine-tune
        dataset_name: HuggingFace dataset name
        output_repo: Where to push the fine-tuned model (e.g. 'username/model-name')
        num_train_samples: Number of training samples to use
        task_type: Type of task - 'chat', 'completion', or 'classification'
        input_field: Field in dataset to use as input
        output_field: Field in dataset to use as output/label
        evaluation_samples: Number of samples to use for evaluation
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        chat_template: Chat template to use (for chat/classification tasks)
        lora_r: LoRA rank
        hf_token: HuggingFace token
    """
    # Get HuggingFace token from environment
    if hf_token is None:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")
    
    # Validate output repository name
    if not output_repo:
        raise ValueError("Please provide an output_repo name (e.g., 'your-username/model-name')")
    
    config = TrainingConfig(
        base_model=base_model,
        dataset_name=dataset_name,
        output_repo=output_repo,
        hf_token=hf_token,
        num_train_samples=num_train_samples,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        evaluation_samples=evaluation_samples,
        task_type=task_type,
        input_field=input_field,
        output_field=output_field,
        chat_template=chat_template,
        lora_r=lora_r
    )
    
    trainer = ModelTrainer()
    results = trainer.train.remote(config)
    
    print(f"Training completed! Results: {results}")
    return results