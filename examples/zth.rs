// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate anyhow;

use std::time::Instant;

use rust_bert::{
    pipelines::{
        common::ModelResource,
        sequence_classification,
        zero_shot_classification::{ZeroShotClassificationConfig, ZeroShotClassificationModel},
    },
    resources::{RemoteResource, ResourceProvider},
};

fn main() -> anyhow::Result<()> {
    // 1. Define Thai-specific resources
    let config_resource = RemoteResource::new(
        "https://huggingface.co/c1ownraid/bert-base-thai-upos-rust/resolve/main/config.json",
        "bert-base-th-config",
    );
    let vocab_resource = RemoteResource::new(
        "https://huggingface.co/c1ownraid/bert-base-thai-upos-rust/resolve/main/vocab.txt",
        "bert-base-th-vocab",
    );

    let model_resource = ModelResource::Torch(Box::new(RemoteResource::new(
        "https://huggingface.co/c1ownraid/bert-base-thai-upos-rust/resolve/main/rust_model.ot",
        "model",
    )));
    // ModelResource::Torch( /)
    let config = ZeroShotClassificationConfig::new(
        rust_bert::pipelines::common::ModelType::DistilBert,
        model_resource,
        config_resource,
        vocab_resource,
        None,
        false,
        false,
        None,
    );

    //    Set-up model
    let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
    let model = ZeroShotClassificationModel::new(config)?;

    let input_sentence = "Who are you voting for in 2020?";
    let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    let candidate_labels = &["politics", "public health", "economy", "sports"];

    let t = Instant::now();
    let output = sequence_classification_model
        .predict_multilabel(
            [input_sentence, input_sequence_2],
            candidate_labels,
            Some(Box::new(|label: &str| {
                format!("This example is about {label}.")
            })),
            128,
        )
        .unwrap();
    println!("***\n{:?}\n", t.elapsed());

    println!("{output:?}");

    Ok(())
}
