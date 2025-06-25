# Complete GRPO Implementation - MIT Supercloud V100 Compatible
# Optimized for V100 GPUs with 16GB VRAM and Supercloud environment

import os
import torch
import torch.nn.functional as F
import re
import json
import math
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass
import logging
import gc

# Handle import errors gracefully
try:
    from datasets import load_dataset, Dataset

    DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: datasets import failed: {e}")
    print("📝 Please run: pip install datasets --upgrade")
    DATASETS_AVAILABLE = False
    # Create dummy classes for type hints
    Dataset = type("Dataset", (), {})
    load_dataset = lambda *args, **kwargs: None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: transformers import failed: {e}")
    print("📝 Please run: pip install transformers --upgrade")
    TRANSFORMERS_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer

    TRL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: trl import failed: {e}")
    print("📝 Please run: pip install trl --upgrade")
    TRL_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: peft import failed: {e}")
    print("📝 Please run: pip install peft --upgrade")
    PEFT_AVAILABLE = False

# V100 compatibility settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU for Supercloud
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

# Set up logging for Supercloud
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_environment_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []

    if not DATASETS_AVAILABLE:
        missing_deps.append("datasets")
    if not TRANSFORMERS_AVAILABLE:
        missing_deps.append("transformers")
    if not TRL_AVAILABLE:
        missing_deps.append("trl")
    if not PEFT_AVAILABLE:
        missing_deps.append("peft")

    if missing_deps:
        print("❌ Missing dependencies:", ", ".join(missing_deps))
        print("\n🔧 To fix the PyArrow/Protobuf conflict, run ONE of these solutions:")
        print("\n1. Quick fix (in your current environment):")
        print("   bash fix_protobuf_conflict.sh")
        print("\n2. Clean environment (recommended):")
        print("   bash create_clean_env.sh")
        print("   conda activate rl_clean")
        print("\n3. Manual fix:")
        print("   pip uninstall -y pyarrow datasets protobuf")
        print("   pip install protobuf==3.20.3 pyarrow==12.0.1 datasets==2.14.0")
        return False

    print("✅ All dependencies available!")
    return True


# =============================================================================
# 1. DATA PREPARATION AND FORMATTING
# =============================================================================


class DataProcessor:
    """Handles complex data preprocessing for GRPO training"""

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are a helpful assistant. Think step by step and provide your reasoning.

Format your response as:
<thinking>
Your step-by-step reasoning here
</thinking>

<answer>
Your final answer here
</answer>"""

    def format_conversation(
        self, question: str, context: str = None
    ) -> List[Dict[str, str]]:
        """Format input as conversation for chat models"""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                }
            )
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def process_gsm8k_dataset(
        self, split: str = "train", max_samples: int = None
    ) -> Dataset:
        """Process GSM8K dataset with proper formatting"""
        dataset = load_dataset("openai/gsm8k", split=split)

        def format_example(example):
            # Extract ground truth answer
            answer_text = example["answer"]
            ground_truth = self._extract_numerical_answer(answer_text)

            return {
                "prompt": self.format_conversation(example["question"]),
                "ground_truth": ground_truth,
                "question_type": "math",
                "difficulty": self._assess_difficulty(example["question"]),
            }

        dataset = dataset.map(format_example)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    def _extract_numerical_answer(self, answer_text: str) -> str:
        """Extract numerical answer from GSM8K format"""
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip().replace(",", "")
        return ""

    def _assess_difficulty(self, question: str) -> str:
        """Simple difficulty assessment based on question complexity"""
        word_count = len(question.split())
        if word_count < 20:
            return "easy"
        elif word_count < 40:
            return "medium"
        else:
            return "hard"


# =============================================================================
# 2. COMPREHENSIVE REWARD FUNCTIONS
# =============================================================================


class RewardFunctionSuite:
    """Collection of reward functions for different aspects of model output"""

    def __init__(self):
        self.transition_words = [
            "first",
            "next",
            "then",
            "because",
            "therefore",
            "finally",
            "however",
            "moreover",
            "furthermore",
            "consequently",
            "thus",
            "in conclusion",
            "as a result",
            "on the other hand",
        ]

    def format_reward(
        self, prompts: List, completions: List[str], **kwargs
    ) -> List[float]:
        """Reward proper XML formatting with thinking and answer tags"""
        rewards = []
        thinking_pattern = r"<thinking>(.*?)</thinking>"
        answer_pattern = r"<answer>(.*?)</answer>"

        for completion in completions:
            reward = 0.0

            # Check for thinking section
            thinking_match = re.search(
                thinking_pattern, completion, re.DOTALL | re.IGNORECASE
            )
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                if len(thinking_content) > 20:  # Substantial reasoning
                    reward += 0.4

                    # Bonus for using transition words
                    lower_content = thinking_content.lower()
                    transition_count = sum(
                        1 for word in self.transition_words if word in lower_content
                    )
                    reward += min(0.2, transition_count * 0.05)

            # Check for answer section
            answer_match = re.search(
                answer_pattern, completion, re.DOTALL | re.IGNORECASE
            )
            if answer_match:
                answer_content = answer_match.group(1).strip()
                if len(answer_content) > 0:
                    reward += 0.3

                    # Bonus for concise but complete answers
                    if 5 <= len(answer_content.split()) <= 20:
                        reward += 0.1

            rewards.append(reward)

        return rewards

    def accuracy_reward(
        self, prompts: List, completions: List[str], ground_truth: List[str], **kwargs
    ) -> List[float]:
        """Reward factual accuracy based on ground truth"""
        rewards = []

        for completion, gt in zip(completions, ground_truth):
            reward = 0.0

            # Extract answer from completion
            predicted_answer = self._extract_answer_from_completion(completion)

            if predicted_answer and gt:
                # Exact match
                if self._normalize_answer(predicted_answer) == self._normalize_answer(
                    gt
                ):
                    reward = 1.0
                # Partial match for numerical answers
                elif self._is_numerical_close(predicted_answer, gt):
                    reward = 0.7
                # Contains correct elements
                elif self._contains_correct_elements(predicted_answer, gt):
                    reward = 0.3

            rewards.append(reward)

        return rewards

    def reasoning_quality_reward(
        self, prompts: List, completions: List[str], **kwargs
    ) -> List[float]:
        """Reward quality of reasoning process"""
        rewards = []

        for completion in completions:
            reward = 0.0

            # Extract thinking section
            thinking_match = re.search(
                r"<thinking>(.*?)</thinking>", completion, re.DOTALL | re.IGNORECASE
            )
            if thinking_match:
                reasoning = thinking_match.group(1).strip()

                # Length-based component (encourages detailed reasoning)
                word_count = len(reasoning.split())
                if word_count >= 30:
                    reward += 0.3
                elif word_count >= 15:
                    reward += 0.2
                elif word_count >= 5:
                    reward += 0.1

                # Structure-based component
                sentences = [s.strip() for s in reasoning.split(".") if s.strip()]
                if len(sentences) >= 3:  # Multi-step reasoning
                    reward += 0.2

                # Mathematical reasoning indicators
                math_indicators = [
                    "calculate",
                    "multiply",
                    "divide",
                    "add",
                    "subtract",
                    "equation",
                    "solve",
                    "formula",
                    "total",
                    "sum",
                ]
                math_count = sum(
                    1 for indicator in math_indicators if indicator in reasoning.lower()
                )
                reward += min(0.2, math_count * 0.05)

                # Logical connectors
                logical_connectors = [
                    "because",
                    "therefore",
                    "since",
                    "given that",
                    "this means",
                    "so",
                    "thus",
                ]
                logic_count = sum(
                    1
                    for connector in logical_connectors
                    if connector in reasoning.lower()
                )
                reward += min(0.2, logic_count * 0.05)

            rewards.append(reward)

        return rewards

    def coherence_reward(
        self, prompts: List, completions: List[str], **kwargs
    ) -> List[float]:
        """Reward overall coherence and consistency"""
        rewards = []

        for completion in completions:
            reward = 0.0

            # Check if thinking and answer are consistent
            thinking_match = re.search(
                r"<thinking>(.*?)</thinking>", completion, re.DOTALL | re.IGNORECASE
            )
            answer_match = re.search(
                r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE
            )

            if thinking_match and answer_match:
                thinking = thinking_match.group(1).strip().lower()
                answer = answer_match.group(1).strip().lower()

                # Extract numbers from both sections
                thinking_numbers = re.findall(r"\d+\.?\d*", thinking)
                answer_numbers = re.findall(r"\d+\.?\d*", answer)

                # Check consistency
                if answer_numbers and thinking_numbers:
                    # Answer number should appear in thinking
                    if any(num in thinking_numbers for num in answer_numbers):
                        reward += 0.3

                # Check for contradiction indicators
                contradictions = ["but", "however", "actually", "wait", "correction"]
                if not any(word in completion.lower() for word in contradictions):
                    reward += 0.2
                else:
                    # Actually, some contradictions might indicate self-correction
                    if (
                        "correction" in completion.lower()
                        or "actually" in completion.lower()
                    ):
                        reward += 0.1  # Partial credit for self-correction

            rewards.append(reward)

        return rewards

    def safety_reward(
        self, prompts: List, completions: List[str], **kwargs
    ) -> List[float]:
        """Penalty for unsafe or inappropriate content"""
        rewards = []

        unsafe_patterns = [
            r"i (can\'t|cannot|won\'t|will not) help",
            r"i don\'t know",
            r"(harmful|dangerous|illegal|unethical)",
            r"i\'m (not sure|uncertain|unsure)",
        ]

        for completion in completions:
            penalty = 0.0
            completion_lower = completion.lower()

            for pattern in unsafe_patterns:
                if re.search(pattern, completion_lower):
                    penalty -= 0.2

            # Bonus for confident, helpful responses
            if not any(
                re.search(pattern, completion_lower) for pattern in unsafe_patterns
            ):
                penalty += 0.1

            rewards.append(penalty)

        return rewards

    def _extract_answer_from_completion(self, completion: str) -> str:
        """Extract the final answer from completion"""
        # Try XML format first
        answer_match = re.search(
            r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Try other common formats
        patterns = [
            r"final answer[:\s]*([^\n]+)",
            r"answer[:\s]*([^\n]+)",
            r"solution[:\s]*([^\n]+)",
            r"result[:\s]*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Last resort: try to find numbers at the end
        numbers = re.findall(r"\d+\.?\d*", completion)
        return numbers[-1] if numbers else ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Remove common formatting
        answer = re.sub(r"[^\w\s\.]", "", answer.lower())
        answer = answer.strip()

        # Handle numerical answers
        try:
            float_val = float(answer)
            return str(float_val)
        except ValueError:
            return answer

    def _is_numerical_close(self, pred: str, gt: str, tolerance: float = 0.01) -> bool:
        """Check if numerical answers are close"""
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            return abs(pred_val - gt_val) / max(abs(gt_val), 1) <= tolerance
        except ValueError:
            return False

    def _contains_correct_elements(self, pred: str, gt: str) -> bool:
        """Check if prediction contains elements of ground truth"""
        pred_words = set(pred.lower().split())
        gt_words = set(gt.lower().split())

        # Check overlap
        overlap = len(pred_words.intersection(gt_words))
        return overlap / max(len(gt_words), 1) > 0.3


# =============================================================================
# 3. ADVANCED TRAINING CONFIGURATION
# =============================================================================


@dataclass
class V100GRPOConfig:
    """V100-optimized configuration for MIT Supercloud"""

    # Base training parameters - V100 optimized
    output_dir: str = "./grpo_outputs"
    run_name: str = "v100_grpo"
    learning_rate: float = 3e-6  # Lower LR for stability
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1  # V100 memory constraint
    gradient_accumulation_steps: int = 16  # Compensate for small batch size

    # GRPO-specific parameters - V100 memory optimized
    num_generations: int = 2  # Reduced for V100 VRAM
    max_prompt_length: int = 256  # Reduced for memory
    max_completion_length: int = 512  # Reduced for memory
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.01  # KL divergence penalty

    # Advanced sampling
    do_sample: bool = True
    repetition_penalty: float = 1.05  # Slight reduction
    length_penalty: float = 1.0

    # V100 memory optimization (NO Flash Attention - not supported on V100)
    use_flash_attention: bool = False  # V100 doesn't support Flash Attention 2
    gradient_checkpointing: bool = True
    bf16: bool = False  # V100 doesn't support bfloat16 efficiently
    fp16: bool = True  # Use fp16 instead for V100

    # Reward function weights
    reward_weights: List[float] = None

    # PEFT configuration - aggressive for V100
    use_peft: bool = True
    lora_r: int = 8  # Reduced rank for V100
    lora_alpha: int = 16  # Reduced alpha
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will be set based on model

    # Logging and saving - frequent for Supercloud
    logging_steps: int = 5
    save_steps: int = 100  # More frequent saves for cluster
    eval_steps: int = 100

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.005

    # V100 specific optimizations
    dataloader_num_workers: int = 2  # Limited for V100
    max_memory_mb: int = 14000  # Reserve 2GB for system
    cleanup_frequency: int = 10  # Cleanup every N steps


# =============================================================================
# 4. COMPLETE TRAINING PIPELINE
# =============================================================================


class V100GRPOTrainer:
    """V100-optimized GRPO training pipeline for MIT Supercloud"""

    def __init__(self, model_name: str, config: V100GRPOConfig):
        self.model_name = model_name
        self.config = config

        self.tokenizer = None
        self.model = None
        self.reward_suite = RewardFunctionSuite()
        self.data_processor = DataProcessor()

        # V100 memory monitoring
        self.peak_memory = 0
        self.memory_cleanup_counter = 0

        self._check_v100_compatibility()
        self._setup_model()
        self._setup_reward_functions()

    def _check_v100_compatibility(self):
        """Check V100 GPU compatibility"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - V100 GPU required")

        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU detected: {gpu_name}")

        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Total GPU memory: {total_memory:.1f} GB")

        if total_memory < 15:  # V100 has 16GB
            logging.warning("GPU memory less than 15GB - may have issues")

    def _setup_model(self):
        """Initialize model and tokenizer with V100 optimizations"""
        logging.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,  # Fast tokenizer for V100
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # V100-optimized model loading
        model_kwargs = {
            "torch_dtype": torch.float16,  # fp16 for V100, not bfloat16
            "low_cpu_mem_usage": True,
            "use_cache": False,  # Required for gradient checkpointing
            "trust_remote_code": True,
        }

        # Load model with careful memory management
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
            self.model = self.model.cuda()  # Explicit GPU placement for V100
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            # Try with even more conservative settings
            model_kwargs["torch_dtype"] = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
            self.model = self.model.half().cuda()  # Convert to fp16 after loading

        # Apply PEFT with V100-optimized settings
        if self.config.use_peft:
            # Auto-detect target modules based on model architecture
            if self.config.target_modules is None:
                if "qwen" in self.model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                elif (
                    "llama" in self.model_name.lower()
                    or "mistral" in self.model_name.lower()
                ):
                    target_modules = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                else:
                    # Conservative default
                    target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = self.config.target_modules

            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                task_type="CAUSAL_LM",
                bias="none",  # No bias for V100 efficiency
            )

            self.model = get_peft_model(self.model, peft_config)
            logging.info(
                f"PEFT applied with r={self.config.lora_r}, targets: {target_modules}"
            )

        # Memory cleanup after model loading
        self._cleanup_memory()
        self._log_memory_usage("After model loading")

    def _cleanup_memory(self):
        """Aggressive memory cleanup for V100"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            logging.info(f"{stage} - Current: {current:.2f}GB, Peak: {peak:.2f}GB")

            if peak > self.peak_memory:
                self.peak_memory = peak

    def _setup_reward_functions(self):
        """Setup reward functions with V100-appropriate weights"""
        self.reward_functions = [
            self.reward_suite.format_reward,
            self.reward_suite.accuracy_reward,
            self.reward_suite.reasoning_quality_reward,
            self.reward_suite.coherence_reward,
            self.reward_suite.safety_reward,
        ]

        # Conservative weights for V100 training
        if self.config.reward_weights is None:
            self.config.reward_weights = [0.25, 0.35, 0.2, 0.15, 0.05]

    def prepare_dataset(
        self, dataset_name: str = "openai/gsm8k", max_samples: int = 500
    ):
        """Prepare training dataset - reduced size for V100"""
        logging.info(f"Preparing dataset: {dataset_name} (max {max_samples} samples)")

        if dataset_name == "openai/gsm8k":
            dataset = self.data_processor.process_gsm8k_dataset("train", max_samples)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported yet")

        logging.info(f"Dataset prepared with {len(dataset)} samples")
        return dataset

    def combined_reward_function(
        self, prompts: List, completions: List[str], **kwargs
    ) -> List[float]:
        """Combine multiple reward functions with weights - V100 optimized"""
        all_rewards = []

        # Memory-efficient computation
        with torch.no_grad():  # Ensure no gradients during reward computation
            for i, reward_func in enumerate(self.reward_functions):
                try:
                    rewards = reward_func(prompts, completions, **kwargs)
                    weighted_rewards = [
                        r * self.config.reward_weights[i] for r in rewards
                    ]
                    all_rewards.append(weighted_rewards)
                except Exception as e:
                    logging.warning(f"Error in reward function {i}: {e}")
                    all_rewards.append([0.0] * len(completions))

        # Sum weighted rewards
        final_rewards = []
        for i in range(len(completions)):
            total_reward = sum(rewards[i] for rewards in all_rewards)
            final_rewards.append(total_reward)

        # Periodic memory cleanup during training
        self.memory_cleanup_counter += 1
        if self.memory_cleanup_counter >= self.config.cleanup_frequency:
            self._cleanup_memory()
            self.memory_cleanup_counter = 0

        return final_rewards

    def train(self, dataset: Dataset):
        """Execute V100-optimized training pipeline"""
        logging.info("🚀 Starting V100-optimized GRPO training...")

        # Configure GRPO training arguments for V100
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            beta=self.config.beta,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            fp16=self.config.fp16,  # Use fp16 for V100
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            remove_unused_columns=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            # V100 specific optimizations
            max_grad_norm=0.5,  # Conservative gradient clipping
            warmup_steps=50,  # Fewer warmup steps
            lr_scheduler_type="cosine",
            save_total_limit=3,  # Limit saved checkpoints for storage
            load_best_model_at_end=True,
            metric_for_best_model="rewards/mean",
            greater_is_better=True,
        )

        # Initialize trainer with V100 optimizations
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_funcs=self.combined_reward_function,
        )

        # Add V100-specific callbacks
        trainer.add_callback(self._create_v100_logging_callback())
        trainer.add_callback(self._create_memory_monitor_callback())

        # Pre-training memory check
        self._log_memory_usage("Before training")

        try:
            # Start training
            trainer.train()

            # Post-training cleanup and save
            self._cleanup_memory()
            trainer.save_model()

            logging.info(
                f"✅ V100 training complete! Model saved to {self.config.output_dir}"
            )
            logging.info(f"Peak memory usage: {self.peak_memory:.2f}GB")

        except Exception as e:
            logging.error(f"Training failed: {e}")
            # Emergency memory cleanup
            self._cleanup_memory()
            raise

    def _create_v100_logging_callback(self):
        """Create V100-specific logging callback"""
        from transformers import TrainerCallback

        class V100LoggingCallback(TrainerCallback):
            def __init__(self, trainer_instance):
                self.trainer_instance = trainer_instance

            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    step = state.global_step

                    # Log rewards
                    if "rewards/mean" in logs:
                        logging.info(
                            f"Step {step}: Mean Reward = {logs['rewards/mean']:.4f}"
                        )
                    if "rewards/std" in logs:
                        logging.info(
                            f"Step {step}: Reward Std = {logs['rewards/std']:.4f}"
                        )

                    # Log memory every few steps
                    if step % 20 == 0:
                        self.trainer_instance._log_memory_usage(f"Step {step}")

            def on_step_end(self, args, state, control, model=None, **kwargs):
                # Periodic cleanup
                if state.global_step % 50 == 0:
                    self.trainer_instance._cleanup_memory()

        return V100LoggingCallback(self)

    def _create_memory_monitor_callback(self):
        """Create memory monitoring callback for V100"""
        from transformers import TrainerCallback

        class MemoryMonitorCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                # Check memory before each step
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / 1024**3
                    if current_mem > 14.5:  # Close to V100 limit
                        logging.warning(f"High memory usage: {current_mem:.2f}GB")
                        gc.collect()
                        torch.cuda.empty_cache()

        return MemoryMonitorCallback()


# =============================================================================
# 5. USAGE EXAMPLE
# =============================================================================


def main_v100():
    """V100-optimized example for MIT Supercloud"""

    # MIT Supercloud V100 Configuration
    config = V100GRPOConfig(
        output_dir="./v100_grpo_model",
        run_name="mit_supercloud_grpo",
        learning_rate=3e-6,  # Conservative for V100
        num_train_epochs=1,
        per_device_train_batch_size=1,  # V100 memory constraint
        gradient_accumulation_steps=16,  # Compensate for small batch
        num_generations=2,  # Reduced for V100 VRAM
        max_prompt_length=256,  # Conservative for memory
        max_completion_length=512,  # Conservative for memory
        temperature=0.7,
        beta=0.01,
        use_peft=True,
        lora_r=8,  # Smaller rank for V100
        lora_alpha=16,
        fp16=True,  # Use fp16 instead of bf16 for V100
        reward_weights=[0.25, 0.35, 0.2, 0.15, 0.05],
        logging_steps=5,
        save_steps=100,
        max_memory_mb=14000,  # Reserve 2GB for system
        cleanup_frequency=10,
    )

    # Use smaller model for V100 compatibility
    model_name = "./Qwen2-0.5B-Instruct"  # Local path to the model

    logging.info("🚀 Starting MIT Supercloud V100 GRPO Training")
    logging.info(f"Model: {model_name}")
    logging.info(f"Max memory target: {config.max_memory_mb}MB")

    # Initialize V100-optimized trainer
    trainer = V100GRPOTrainer(model_name=model_name, config=config)

    # Prepare smaller dataset for V100
    dataset = trainer.prepare_dataset("openai/gsm8k", max_samples=200)  # Small dataset

    # Train with V100 optimizations
    trainer.train(dataset)

    logging.info("🎉 MIT Supercloud V100 GRPO training completed!")


# Supercloud job submission script helper
def create_supercloud_job_script():
    """Create SLURM job script for MIT Supercloud"""
    job_script = """#!/bin/bash
#SBATCH --job-name=grpo_v100
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Load modules
module load anaconda/2023a
module load cuda/11.8

# Activate environment
source activate your_env_name

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training
cd $SLURM_SUBMIT_DIR
python v100_grpo_training.py

echo "Job completed at $(date)"
"""

    with open("submit_grpo.sh", "w") as f:
        f.write(job_script)

    logging.info("Created submit_grpo.sh for MIT Supercloud")
    logging.info("Submit with: sbatch submit_grpo.sh")


# MIT Supercloud environment setup
def setup_supercloud_environment():
    """Setup environment for MIT Supercloud"""

    setup_commands = """
# MIT Supercloud Setup Commands
# Run these on the login node before submitting job

# 1. Load required modules
module load anaconda/2023a
module load cuda/11.8

# 2. Create conda environment
conda create -n grpo_env python=3.9 -y
conda activate grpo_env

# 3. Install PyTorch for V100 (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 4. Install transformers and TRL
pip install transformers==4.36.0
pip install trl==0.7.4
pip install datasets==2.14.0
pip install peft==0.6.0
pip install accelerate==0.24.0

# 5. Install additional dependencies
pip install numpy pandas scikit-learn
pip install wandb  # Optional for logging

# 6. Test installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
"""

    with open("setup_supercloud.sh", "w") as f:
        f.write(setup_commands)

    logging.info("Created setup_supercloud.sh")
    logging.info("Run: bash setup_supercloud.sh")


if __name__ == "__main__":
    # Check dependencies first
    if not check_environment_dependencies():
        print("\n❌ Cannot proceed with missing dependencies.")
        print("Please fix the environment and try again.")
        exit(1)

    # Create helper scripts
    create_supercloud_job_script()
    setup_supercloud_environment()

    # Run V100 training
    main_v100()


class V100GRPOEvaluator:
    """V100-optimized evaluator for GRPO-trained models"""

    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.load_model()

    def load_model(self):
        """Load trained model for evaluation with V100 optimizations"""
        logging.info(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=True, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # V100 optimization
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.cuda()
        self.model.eval()

        # Memory cleanup after loading
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate_on_test_set(self, test_dataset: Dataset, num_samples: int = 50):
        """Evaluate model performance on test set - V100 optimized"""
        logging.info(f"Evaluating on {num_samples} samples")
        results = []

        for i, example in enumerate(test_dataset.select(range(num_samples))):
            if i % 10 == 0:
                logging.info(f"Processing sample {i}/{num_samples}")
                # Periodic memory cleanup
                gc.collect()
                torch.cuda.empty_cache()

            # Generate response with V100 settings
            try:
                inputs = self.tokenizer.apply_chat_template(
                    example["prompt"], return_tensors="pt", add_generation_prompt=True
                ).cuda()
            except:
                # Fallback for models without chat template
                prompt_text = example["prompt"][-1]["content"]
                inputs = self.tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=256
                ).input_ids.cuda()

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,  # Conservative for V100
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # For evaluation
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            )

            # Evaluate response
            reward_suite = RewardFunctionSuite()

            format_score = reward_suite.format_reward([""], [response])[0]
            accuracy_score = reward_suite.accuracy_reward(
                [""], [response], [example["ground_truth"]]
            )[0]
            reasoning_score = reward_suite.reasoning_quality_reward([""], [response])[0]

            results.append(
                {
                    "sample_id": i,
                    "question": (
                        example["prompt"][-1]["content"]
                        if isinstance(example["prompt"], list)
                        else example["prompt"]
                    ),
                    "response": response,
                    "ground_truth": example["ground_truth"],
                    "format_score": format_score,
                    "accuracy_score": accuracy_score,
                    "reasoning_score": reasoning_score,
                    "total_score": format_score + accuracy_score + reasoning_score,
                }
            )

        # Calculate averages
        avg_format = np.mean([r["format_score"] for r in results])
        avg_accuracy = np.mean([r["accuracy_score"] for r in results])
        avg_reasoning = np.mean([r["reasoning_score"] for r in results])
        avg_total = np.mean([r["total_score"] for r in results])

        logging.info(f"V100 Evaluation Results on {num_samples} samples:")
        logging.info(f"Average Format Score: {avg_format:.3f}")
        logging.info(f"Average Accuracy Score: {avg_accuracy:.3f}")
        logging.info(f"Average Reasoning Score: {avg_reasoning:.3f}")
        logging.info(f"Average Total Score: {avg_total:.3f}")

        return results


# MIT Supercloud specific utilities
def check_supercloud_resources():
    """Check available resources on MIT Supercloud"""
    logging.info("=== MIT Supercloud Resource Check ===")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"✅ GPU: {gpu_name}")
        logging.info(f"✅ Total VRAM: {total_memory:.1f} GB")

        if "V100" not in gpu_name:
            logging.warning("⚠️  Not a V100 GPU - may need different settings")
    else:
        logging.error("❌ No GPU available")

    # Check CUDA version
    logging.info(f"✅ CUDA Version: {torch.version.cuda}")

    # Check PyTorch version
    logging.info(f"✅ PyTorch Version: {torch.__version__}")

    # Check available CPU cores
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    logging.info(f"✅ CPU Cores: {cpu_count}")

    # Check memory
    import psutil

    memory = psutil.virtual_memory()
    logging.info(f"✅ System RAM: {memory.total / 1024**3:.1f} GB")

    logging.info("=== Resource Check Complete ===")


# Example usage for MIT Supercloud:
"""
# On MIT Supercloud login node:
1. Run: bash setup_supercloud.sh
2. Submit job: sbatch submit_grpo.sh
3. Monitor: squeue -u $USER
4. Check output: tail -f grpo_*.out

# For interactive testing:
srun --partition=xeon-g6-volta --gres=gpu:volta:1 --cpus-per-task=4 --mem=32G --time=1:00:00 --pty bash
module load anaconda/2023a cuda/11.8
conda activate grpo_env
python v100_grpo_training.py
"""
