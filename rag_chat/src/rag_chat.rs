use std::error::Error;
use std::io;
use std::io::Write;
use argh;
use argh::FromArgs;
use rten::Model;
use rten_generate::Generator;
use rten_text::Tokenizer;

mod helpers;
use crate::helpers::{ChatConfig, do_one_iteration, generator_from_model};

#[derive(FromArgs)]
#[argh(description="cli args")]
pub(crate) struct Args {
    #[argh(positional)]
    pub(crate) model: String,
    #[argh(positional)]
    pub(crate) tokenizer_config: String,
}

fn do_prompt() -> () {
    print!("> ");
    let _ = io::stdout().flush();
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
    let mut generator = Generator::from_model(&model)?;
    generator = generator_from_model(generator, &tokenizer, config.top_k, config.temperature);

    loop {
        do_prompt();
        do_one_iteration(&mut generator, &tokenizer)?;
    }
}
