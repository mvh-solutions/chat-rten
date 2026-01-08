mod helpers;

use std::error::Error;
use std::io;
use std::io::prelude::*;

use argh;
use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::Tokenizer;

use helpers::{MessageChunk, encode_message, Args};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = argh::from_env();
    args.temperature = args.temperature.max(0.);

    let model = unsafe { Model::load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    // From `chat_template` in tokenizer_config.json.
    let prompt_tokens = encode_message(
        &tokenizer,
        &[
            MessageChunk::Token(im_start_token),
            MessageChunk::Text("system\nYou are a helpful assistant."),
            MessageChunk::Token(im_end_token),
        ],
    )?;

    // From Qwen2's `generation_config.json`
    let top_k = 20;

    let mut generator = Generator::from_model(&model)?
        .with_prompt(&prompt_tokens)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(args.temperature))
        .with_sampler(Multinomial::new());

    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut user_input = String::new();
        let n_read = io::stdin().read_line(&mut user_input)?;
        if n_read == 0 {
            // EOF
            break;
        }

        // From `chat_template` in tokenizer_config.json.
        let token_ids = encode_message(
            &tokenizer,
            &[
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("user\n"),
                MessageChunk::Text(&user_input),
                MessageChunk::Token(im_end_token),
                MessageChunk::Text("\n"),
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("assistant\n"),
            ],
        )?;

        generator.append_prompt(&token_ids);

        let decoder = generator
            .by_ref()
            .stop_on_tokens(&end_of_turn_tokens)
            .decode(&tokenizer);
        for token in decoder {
            let token = token?;
            print!("{}", token);
            let _ = io::stdout().flush();
        }

        println!();
    }

    Ok(())
}
