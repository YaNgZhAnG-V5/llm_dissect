import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import mmengine
import torch
from datasets import load_dataset
from mmengine.runner import set_random_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from dissect.models import build_model_and_tokenizer
from dissect.utils import get_cuda_visible_devices


class IdentityLlamaAttention(nn.Module):
    """
    A class that replaces the LlamaAttention module with an identity layer.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        return hidden_states, None, past_key_value


class IdentityLlamaMLP(nn.Module):
    """
    A class that replaces the LlamaMLP module with an identity layer.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Prompter(object):
    def __init__(self):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context."
                " Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
                "\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            ),
            "response_split": "### Response:",
        }

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def parse_args():
    parser = ArgumentParser("Test pruned models")
    parser.add_argument("--config", default="./configs/llama3_8b_per_attn_perp.yaml", help="Path to config file.")
    parser.add_argument(
        "--pruning-dir",
        "-p",
        # required=True,
        default="workdirs/rebuttal/perplexity/wikitext/llama3_8b",
        help="Directory where the pruning results were stored. "
        'It should contain a sub-directory "pruning_masks/" storing the pruning masks.',
    )
    parser.add_argument(
        "--work-dir", "-w", default="workdirs/prune_vicuna/", help="Working directory to save the output files."
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="micro batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--cutoff_len", type=int, default=256, help="cutoff length")
    parser.add_argument("--val_set_size", type=int, default=2000, help="validation set size")

    # Lora Configuration
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
        help="lora target modules",
    )

    # llm hyperparameters
    parser.add_argument(
        "--train_on_inputs",
        default=False,
        action="store_true",
        help="Train on inputs. If False, masks out inputs in loss",
    )
    parser.add_argument("--add_eos_token", default=False, action="store_true")
    parser.add_argument(
        "--group_by_length", default=False, action="store_true", help="faster, but produces an odd training loss curve"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(42)
    work_dir = args.work_dir
    sparsity_table_save_dir = osp.join(work_dir, "sparsity_tables")
    mmengine.mkdir_or_exist(work_dir)
    mmengine.mkdir_or_exist(sparsity_table_save_dir)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(work_dir, f"{time_stamp}.log"),
    )

    # Pre-process config
    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.pruning_dir = args.pruning_dir
    cfg.work_dir = args.work_dir
    if cfg.test_dataset.split == "train":
        logger.warning("cfg.test_dataset.split is 'train'. Automatically override it to 'test'.")
        cfg.test_dataset["split"] = "test"
    if not cfg.test_dataset.use_label:
        logger.warning(
            "Testing models requires ground truth labels, but cfg.test_dataset.use_label: "
            f"{cfg.test_dataset.use_label}. This config value will be automatically set to True."
        )
        cfg.test_dataset.use_label = True
    logger.info("Using config:\n" + "=" * 60 + f"\n{cfg.pretty_text}\n" + "=" * 60)
    cfg.dump(osp.join(work_dir, f"{time_stamp}_{osp.splitext(osp.basename(cfg.filename))[0]}.yaml"))

    cuda_visible_devices = get_cuda_visible_devices()
    if len(cuda_visible_devices) > 1:
        logger.info(
            f"Running multi-gpu inference on GPUs: {cuda_visible_devices}. The argument: "
            f"--gpu-id {args.gpu_id} is automatically set to 0, indicating that the inference starts from "
            f"GPU 0."
        )
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    model, tokenizer = build_model_and_tokenizer(cfg.model, device=device)

    # load pruned model and apply identity layers
    sparsity = 0.2
    mask_path = osp.join(
        args.pruning_dir, "pruning_masks", f'sparsity_{str(sparsity).replace(".", "_")}_pruning_masks.pth'
    )
    mask_state_dict = torch.load(mask_path, map_location=device)

    for layer_name in mask_state_dict.keys():
        layer = model.get_submodule(".".join(layer_name.split(".")[:-1]))
        assert isinstance(layer, LlamaAttention) or isinstance(layer, LlamaMLP)

        # Replace the submodule with the identity layer
        parent_module = model.get_submodule(".".join(layer_name.split(".")[:-2]))
        if isinstance(layer, LlamaAttention):
            setattr(parent_module, layer_name.split(".")[-2], IdentityLlamaAttention())
        else:
            setattr(parent_module, layer_name.split(".")[-2], IdentityLlamaMLP())

    # perform lora
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # get data
    data_path = "yahma/alpaca-cleaned"
    data = load_dataset(data_path)
    prompter = Prompter()

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):

        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"] if "input" in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    # get trainer
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        logging_first_step=True,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=200,
        output_dir=args.work_dir,
        save_total_limit=20,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=None,
        group_by_length=args.group_by_length,
        report_to=None,
        run_name=None,
        # metric_for_best_model="{}_loss".format(args.data_path),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=None, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()


if __name__ == "__main__":
    main()
