mod helpers;
mod rag_chat_lib;

use std::error::Error;

use argh;
use argh::FromArgs;
use rten::Model;
use rten_generate::Generator;
use rten_text::Tokenizer;

use crate::helpers::{ChatConfig, do_one_iteration, generator_from_model};

#[derive(FromArgs)]
#[argh(description="cli args")]
pub(crate) struct Args {
    /// input model
    #[argh(positional)]
    pub(crate) model: String,

    /// tokenizer.json file
    #[argh(positional)]
    pub(crate) tokenizer_config: String,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let config = ChatConfig {
        model_path: args.model,
        tokenizer_path: args.tokenizer_config,
        temperature: 0.4,
        top_k: 20
    };

    let model = unsafe { Model::load_mmap(config.model_path) }?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer_path)?;

    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    let mut generator = Generator::from_model(&model)?;
    generator = generator_from_model(generator, &tokenizer, config.top_k, config.temperature);

    loop {
        do_one_iteration(&mut generator, &tokenizer, &end_of_turn_tokens)?;
    }
}
