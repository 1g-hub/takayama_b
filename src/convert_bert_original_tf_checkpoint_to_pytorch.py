# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""


import argparse
import os
import torch
import transformers as tf
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    print(tf.__version__)
    # parser = argparse.ArgumentParser()
    # # Required parameters
    # parser.add_argument(
    #     "--tf_checkpoint_path", default="../models/bert/hottoSNS-bert/model.ckpt-1000000.data-00000-of-00001", type=str, required=True, help="Path to the TensorFlow checkpoint path."
    # )
    # parser.add_argument(
    #     "--bert_config_file",
    #     default="../models/bert/hottoSNS-bert/bert_config.json",
    #     type=str,
    #     required=True,
    #     help="The config json file corresponding to the pre-trained BERT model. \n"
    #     "This specifies the model architecture.",
    # )
    # parser.add_argument(
    #     "--pytorch_dump_path", default="../models/bert/hottoSNS-bert-pytorch", type=str, required=True, help="Path to the output PyTorch model."
    # )
    # args = parser.parse_args()

    model_path = "../models/bert/hottoSNS-bert/model.ckpt-1000000"
    config_path = "../models/bert/hottoSNS-bert/bert_config.json"
    export_path = "../models/bert/hottoSNS-bert-pytorch/hottoSNS-bert-pytorch.bin"
    # convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path=model_path, bert_config_file=config_path, pytorch_dump_path=export_path)